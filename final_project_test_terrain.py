"""
Evaluate a selected controller/body checkpoint on chosen terrains.

By default this script uses both controller genes and body genes from the
checkpoint path. Pass --controller to choose how the controller slice is decoded.
Pass --use-archive-body to instead select the body exactly as requested from the
root archive files:

    best_idx = np.argmax(archive_fitness)
    best_body = archive_body[best_idx]

Examples
--------
    # Challenge3 MLP checkpoint
    python final_project_test_terrain.py --controller mlp --controller-checkpoint results/AntHill-v0/single --terrains flat hill

    # Final-project SO2 checkpoint
    python final_project_test_terrain.py --controller so2 --controller-checkpoint results/final_project_so2_climb --terrains flat,ice,hill,eval

    python final_project_test_terrain.py --terrains all --use-archive-body
    python final_project_test_terrain.py --terrains eval --record-video
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import zipfile
from os.path import join
from typing import Iterable

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl" if platform.system() == "Linux" else "glfw")

import evorob.world  # registers FlatEnv-v0 / IceEnv-v0 / HillEnv-v0 / EvalEnv-v0
import gymnasium as gym

from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.eval_world import EvalWorld
from final_project_body_search import gene_to_length
from final_project_train import FinalWorld


ROOT_DIR = get_project_root()

ARCHIVE_BODY_PATH = join(ROOT_DIR, "archive_body.npy")
ARCHIVE_FITNESS_PATH = join(ROOT_DIR, "archive_fitness.npy")
CONTROLLER_CHECKPOINT = join(ROOT_DIR, "results", "AntHill-v0/single")
# CONTROLLER_CHECKPOINT = join(ROOT_DIR, "results", "final_project_so2_climb")
DEFAULT_CONTROLLER = "mlp"
OUTPUT_DIR = join(ROOT_DIR, "evaluation_output", "terrain_test")

SEED = 0
N_EPISODES = 10
MAX_STEPS = 1000

TRAINING_TERRAINS = {
    "flat": ("FlatEnv-v0", "flat_world_file"),
    "ice": ("IceEnv-v0", "ice_world_file"),
    "hill": ("HillEnv-v0", "hill_world_file"),
}
TERRAIN_ALIASES = {
    "flatenv-v0": "flat",
    "iceenv-v0": "ice",
    "hillenv-v0": "hill",
    "evalenv-v0": "eval",
    "evaluation": "eval",
    "test": "eval",
}


def neutral_step_score(info: dict) -> float:
    """Final-project neutral score used by the compatibility/eval scripts."""
    return (
        float(info.get("healthy_reward", 1.0))
        + float(info.get("x_position", 0.0))
        - float(info.get("ctrl_cost", 0.0))
        - float(info.get("cfrc_cost", 0.0))
    )


def split_terrains(values: Iterable[str]) -> list[str]:
    """Accept either '--terrains flat ice' or '--terrains flat,ice'."""
    terrains: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            if item:
                terrains.append(item)

    if not terrains or terrains == ["all"]:
        terrains = ["flat", "ice", "hill", "eval"]

    normalized: list[str] = []
    for terrain in terrains:
        terrain = TERRAIN_ALIASES.get(terrain, terrain)
        if terrain == "all":
            normalized.extend(["flat", "ice", "hill", "eval"])
        elif terrain in (*TRAINING_TERRAINS.keys(), "eval"):
            normalized.append(terrain)
        else:
            valid = ", ".join([*TRAINING_TERRAINS.keys(), "eval", "all"])
            raise argparse.ArgumentTypeError(
                f"Unknown terrain '{terrain}'. Valid terrains: {valid}."
            )

    # Preserve user order but skip duplicates.
    seen = set()
    return [t for t in normalized if not (t in seen or seen.add(t))]


def load_archive_best_body(
    archive_body_path: str,
    archive_fitness_path: str,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Load archive arrays and return (best_idx, best_fitness, best_body, all_fitness)."""
    archive_body = np.load(archive_body_path, allow_pickle=True)
    archive_fitness = np.load(archive_fitness_path, allow_pickle=True)

    if archive_body.ndim != 2:
        raise ValueError(f"archive_body must be 2-D, got shape {archive_body.shape}")
    if archive_body.shape[1] != 8:
        raise ValueError(
            "Expected 8 body genes per archived body, "
            f"got shape {archive_body.shape}."
        )

    fitness = np.asarray(archive_fitness, dtype=float)
    if fitness.ndim == 1:
        selection_values = fitness
    elif fitness.shape[0] == archive_body.shape[0]:
        # Fallback for a multi-objective archive: choose the best scalar sum.
        selection_values = fitness.reshape(fitness.shape[0], -1).sum(axis=1)
    else:
        raise ValueError(
            "archive_fitness first dimension must match archive_body. "
            f"Got body {archive_body.shape}, fitness {fitness.shape}."
        )

    if selection_values.shape[0] != archive_body.shape[0]:
        raise ValueError(
            "archive_fitness length must match archive_body rows. "
            f"Got body {archive_body.shape[0]}, fitness {selection_values.shape[0]}."
        )

    # Keep the requested selection rule explicit.
    best_idx = int(np.argmax(selection_values))
    best_body = np.asarray(archive_body[best_idx], dtype=float)
    best_fitness = float(selection_values[best_idx])
    return best_idx, best_fitness, best_body, fitness


def load_checkpoint_genotype(source: str) -> tuple[np.ndarray, str]:
    """Load x_best.npy from a direct .npy path, checkpoint directory, or zip."""
    source = os.path.expanduser(source)

    if os.path.isfile(source) and source.endswith(".npy"):
        return np.load(source, allow_pickle=True), source

    if os.path.isfile(source) and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            x_best_names = [name for name in zf.namelist() if name.endswith("/x_best.npy")]
            if not x_best_names:
                raise FileNotFoundError(f"No x_best.npy found inside zip: {source}")

            def zip_checkpoint_key(name: str) -> tuple[int, str]:
                parts = name.strip("/").split("/")
                parent = parts[-2] if len(parts) >= 2 else ""
                return (int(parent) if parent.isdigit() else -1, name)

            best_name = max(x_best_names, key=zip_checkpoint_key)
            with zf.open(best_name) as f:
                return np.load(io.BytesIO(f.read()), allow_pickle=True), f"{source}:{best_name}"

    if os.path.isdir(source):
        # Prefer the explicitly named directory, then fall back to latest numeric checkpoint.
        candidates = [join(source, "x_best.npy")]
        last_gen = get_last_checkpoint_dir(source)
        if last_gen is not None:
            candidates.append(join(last_gen, "x_best.npy"))
        for path in candidates:
            if os.path.isfile(path):
                return np.load(path, allow_pickle=True), path

    raise FileNotFoundError(f"Could not find x_best.npy in: {source}")


def make_controller(controller_name: str, hidden_size: int | None):
    """Instantiate the requested controller and return (controller, resolved_hidden)."""
    controller_name = controller_name.lower()
    if controller_name == "so2":
        from evorob.world.robot.controllers.so2 import SO2Controller

        resolved_hidden = 8 if hidden_size is None else hidden_size
        return SO2Controller(input_size=27, output_size=8, hidden_size=resolved_hidden), resolved_hidden

    if controller_name == "mlp":
        from evorob.world.robot.controllers.mlp import NeuralNetworkController

        resolved_hidden = 16 if hidden_size is None else hidden_size
        return NeuralNetworkController(
            input_size=27,
            output_size=8,
            hidden_size=resolved_hidden,
        ), resolved_hidden

    raise ValueError(f"Unsupported controller: {controller_name}")


def resolve_controller_param_scale(controller_name: str, scale: str) -> float:
    """Return the multiplier applied when loading controller genes into phenotype."""
    if scale == "auto":
        # Challenge3 MLP checkpoints were trained with *0.1; final-project SO2
        # evaluation currently matches final_project_test.py when loaded raw.
        return 0.1 if controller_name == "mlp" else 1.0
    return float(scale)


def build_world(args: argparse.Namespace) -> tuple[FinalWorld, dict, np.ndarray]:
    """Create FinalWorld using the requested controller and selected body source."""
    world = FinalWorld()
    controller, resolved_hidden = make_controller(args.controller, args.hidden_size)
    world.controller = controller
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params

    checkpoint_genotype, checkpoint_path = load_checkpoint_genotype(args.controller_checkpoint)
    if checkpoint_genotype.size < world.n_params:
        raise ValueError(
            f"Checkpoint has {checkpoint_genotype.size} genes, "
            f"but controller={args.controller} hidden_size={resolved_hidden} expects "
            f"{world.n_params} ({world.n_weights} controller + "
            f"{world.n_body_params} body)."
        )

    controller_genes = np.asarray(checkpoint_genotype[: world.n_weights], dtype=float)
    controller_param_scale = resolve_controller_param_scale(
        args.controller,
        args.controller_param_scale,
    )
    metadata = {
        "body_source": "archive" if args.use_archive_body else "checkpoint",
        "controller": args.controller,
        "controller_hidden_size": int(resolved_hidden),
        "controller_checkpoint": os.path.abspath(args.controller_checkpoint),
        "controller_genotype_path": os.path.abspath(checkpoint_path),
        "controller_params": int(world.n_weights),
        "controller_param_scale": float(controller_param_scale),
        "body_params": int(world.n_body_params),
        "genotype_params": int(world.n_params),
    }

    if args.use_archive_body:
        best_idx, best_fitness, selected_body, archive_fitness = load_archive_best_body(
            args.archive_body,
            args.archive_fitness,
        )
        metadata.update(
            {
                "archive_body_path": os.path.abspath(args.archive_body),
                "archive_fitness_path": os.path.abspath(args.archive_fitness),
                "archive_size": int(np.asarray(archive_fitness).shape[0]),
                "archive_best_idx": int(best_idx),
                "archive_best_fitness": float(best_fitness),
            }
        )
    else:
        selected_body = np.asarray(
            checkpoint_genotype[world.n_weights : world.n_params], dtype=float
        )
        if selected_body.size != world.n_body_params:
            raise ValueError(
                f"Checkpoint body slice has {selected_body.size} genes, "
                f"expected {world.n_body_params}."
            )

    genotype = np.concatenate([controller_genes, selected_body])
    if genotype.size != world.n_params:
        raise ValueError(
            f"Combined genotype has {genotype.size} params, expected {world.n_params}."
        )

    # FinalWorld.update_robot_xml() regenerates Robot.xml from the full genotype.
    # It also calls FinalWorld.geno2pheno(), which always applies its own *0.1
    # controller scaling. Reload here with the user-selected scale so controller
    # loading is explicit and tunable for SO2, Challenge3 MLP, or future modes.
    world.update_robot_xml(genotype)
    world.controller.geno2pheno(controller_genes * controller_param_scale)

    metadata.update(
        {
            "selected_body_raw": selected_body.tolist(),
            "selected_body_lengths_m": gene_to_length(selected_body).tolist(),
            "controller_param_loading": "explicit_reload_after_body_xml_generation",
        }
    )
    return world, metadata, genotype


def make_eval_world(training_world: FinalWorld) -> EvalWorld:
    """Inject the generated Robot.xml into the hidden-style eval terrain."""
    eval_world = EvalWorld()
    robot_xml = join(training_world.temp_dir.name, "Robot.xml")
    eval_world.update_robot_xml(robot_xml)
    return eval_world


def run_episodes(
    *,
    controller,
    sensor_fn,
    env_id: str,
    robot_path: str,
    n_episodes: int,
    max_steps: int,
    seed: int,
    use_neutral_reward: bool,
) -> list[dict]:
    """Run one terrain and return per-episode score dictionaries."""
    rng = np.random.default_rng(seed)
    env = gym.make(env_id, robot_path=robot_path, max_episode_steps=max_steps)
    rows: list[dict] = []

    for episode in range(n_episodes):
        controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        total = 0.0
        episode_len = 0
        done = False
        while not done:
            ctrl_obs = sensor_fn(obs) if sensor_fn is not None else obs
            action = controller.get_action(ctrl_obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)
            total += neutral_step_score(info) if use_neutral_reward else float(reward)
            episode_len += 1
            done = terminated or truncated
        rows.append({"episode": episode + 1, "score": float(total), "steps": episode_len})

    env.close()
    return rows


def record_video(
    *,
    controller,
    sensor_fn,
    env_id: str,
    robot_path: str,
    out_path: str,
    max_steps: int,
    seed: int,
) -> None:
    """Record one rollout for a terrain when imageio/rendering are available."""
    try:
        import imageio

        env = gym.make(
            env_id,
            robot_path=robot_path,
            render_mode="rgb_array",
            max_episode_steps=max_steps,
        )
        controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames = []
        for _ in range(max_steps):
            frames.append(env.render())
            ctrl_obs = sensor_fn(obs) if sensor_fn is not None else obs
            action = controller.get_action(ctrl_obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved: {out_path}")
    except Exception as exc:  # pragma: no cover - optional rendering dependency
        print(f"  Video skipped for {env_id}: {exc}")


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "best": float(arr.max()),
        "worst": float(arr.min()),
    }


def write_outputs(
    *,
    output_dir: str,
    metadata: dict,
    terrain_results: dict[str, list[dict]],
    genotype: np.ndarray,
    robot_xml: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    np.save(join(output_dir, f"x_test_terrain_{metadata['controller']}.npy"), genotype)
    shutil.copy2(robot_xml, join(output_dir, "Robot.xml"))

    rows_path = join(output_dir, "terrain_episode_scores.csv")
    with open(rows_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["terrain", "episode", "score", "steps"])
        writer.writeheader()
        for terrain, rows in terrain_results.items():
            for row in rows:
                writer.writerow({"terrain": terrain, **row})

    summary = dict(metadata)
    summary["terrains"] = {
        terrain: summarize([row["score"] for row in rows])
        for terrain, rows in terrain_results.items()
    }
    summary["score_csv"] = rows_path
    with open(join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    score_path = join(output_dir, "terrain_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("Selected body + selected controller terrain evaluation\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Body source         : {metadata['body_source']}\n")
        if metadata["body_source"] == "archive":
            f.write(f"Archive best idx    : {metadata['archive_best_idx']}\n")
            f.write(f"Archive best fitness: {metadata['archive_best_fitness']:.6f}\n")
        f.write(
            f"Controller          : {metadata['controller']} "
            f"hidden={metadata['controller_hidden_size']}\n"
        )
        f.write(f"Controller genotype : {metadata['controller_genotype_path']}\n")
        f.write(f"Controller scale    : {metadata['controller_param_scale']}\n")
        f.write(f"Controller loading  : {metadata['controller_param_loading']}\n")
        f.write(f"Body raw genes      : {metadata['selected_body_raw']}\n")
        f.write(f"Body lengths [m]    : {metadata['selected_body_lengths_m']}\n\n")
        f.write(f"{'Terrain':<8} {'Mean':>12} {'Std':>12} {'Best':>12} {'Worst':>12}\n")
        f.write("-" * 72 + "\n")
        for terrain, rows in terrain_results.items():
            stats = summarize([row["score"] for row in rows])
            f.write(
                f"{terrain:<8} {stats['mean']:12.2f} {stats['std']:12.2f} "
                f"{stats['best']:12.2f} {stats['worst']:12.2f}\n"
            )
    print(f"Saved scores: {score_path}")
    print(f"Saved CSV   : {rows_path}")
    print(f"Saved summary: {join(output_dir, 'summary.json')}")
    print(f"Saved genotype/Robot.xml in: {output_dir}")


def evaluate(args: argparse.Namespace) -> dict[str, list[dict]]:
    terrains = split_terrains(args.terrains)
    os.makedirs(args.output_dir, exist_ok=True)

    world, metadata, genotype = build_world(args)
    robot_xml = join(world.temp_dir.name, "Robot.xml")

    print(f"Body source: {metadata['body_source']}")
    if metadata["body_source"] == "archive":
        print(f"  best_idx={metadata['archive_best_idx']}")
        print(f"  archive_fitness={metadata['archive_best_fitness']:.6f}")
    print(f"  body raw={np.asarray(metadata['selected_body_raw'])}")
    print(f"  body lengths [m]={np.asarray(metadata['selected_body_lengths_m'])}")
    print(
        f"Controller: {metadata['controller']} hidden={metadata['controller_hidden_size']} "
        f"from {metadata['controller_genotype_path']}"
    )
    print(
        f"Controller loading: {metadata['controller_param_loading']} "
        f"scale={metadata['controller_param_scale']}"
    )
    print(
        f"Running {args.n_episodes} episode(s), max_steps={args.steps}, "
        f"reward={'neutral' if args.neutral_reward else 'env'}\n"
    )

    eval_world = make_eval_world(world) if "eval" in terrains else None
    terrain_results: dict[str, list[dict]] = {}

    for terrain in terrains:
        if terrain == "eval":
            assert eval_world is not None
            env_id = "EvalEnv-v0"
            robot_path = eval_world.world_file
        else:
            env_id, world_file_attr = TRAINING_TERRAINS[terrain]
            robot_path = getattr(world, world_file_attr)

        print(f"{terrain}: {env_id}")
        rows = run_episodes(
            controller=world.controller,
            sensor_fn=world.sensor_fn,
            env_id=env_id,
            robot_path=robot_path,
            n_episodes=args.n_episodes,
            max_steps=args.steps,
            seed=args.seed,
            use_neutral_reward=args.neutral_reward,
        )
        terrain_results[terrain] = rows
        stats = summarize([row["score"] for row in rows])
        for row in rows:
            print(
                f"  episode {row['episode']:3d}/{args.n_episodes}: "
                f"score={row['score']:10.2f} steps={row['steps']}"
            )
        print(
            f"  {terrain} summary: mean={stats['mean']:.2f} ± {stats['std']:.2f} "
            f"best={stats['best']:.2f} worst={stats['worst']:.2f}\n"
        )

        if args.record_video:
            record_video(
                controller=world.controller,
                sensor_fn=world.sensor_fn,
                env_id=env_id,
                robot_path=robot_path,
                out_path=join(args.output_dir, f"evaluation_{terrain}.mp4"),
                max_steps=args.steps,
                seed=args.seed,
            )

    write_outputs(
        output_dir=args.output_dir,
        metadata=metadata,
        terrain_results=terrain_results,
        genotype=genotype,
        robot_xml=robot_xml,
    )
    return terrain_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a selected controller/body checkpoint on selected terrains. "
            "By default, use the checkpoint body too; pass --use-archive-body "
            "to use archive_body[np.argmax(archive_fitness)]."
        )
    )
    parser.add_argument(
        "--terrains",
        nargs="+",
        default=["flat", "ice", "hill"],
        help="Terrains to run: flat, ice, hill, eval, all. Accepts spaces or commas.",
    )
    parser.add_argument(
        "--use-archive-body",
        action="store_true",
        help="Use archive_body[np.argmax(archive_fitness)] instead of the checkpoint body.",
    )
    parser.add_argument("--archive-body", default=ARCHIVE_BODY_PATH)
    parser.add_argument("--archive-fitness", default=ARCHIVE_FITNESS_PATH)
    parser.add_argument(
        "--controller",
        choices=["mlp", "so2"],
        default=DEFAULT_CONTROLLER,
        help="Controller decoder to use for the checkpoint controller slice.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size for MLP; SO2 accepts it for compatibility (default: mlp=16, so2=8).",
    )
    parser.add_argument(
        "--controller-param-scale",
        default="auto",
        help="Multiplier applied when loading controller genes. auto = 0.1 for mlp, 1.0 for so2.",
    )
    parser.add_argument("--controller-checkpoint", default=CONTROLLER_CHECKPOINT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--env-reward",
        dest="neutral_reward",
        action="store_false",
        help="Use the terrain environment reward instead of neutral final-project score.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record one video per selected terrain.",
    )
    parser.set_defaults(neutral_reward=True)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
