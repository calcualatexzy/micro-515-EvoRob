"""
Body-only search for the MICRO-515 final project.

This script keeps the trained MLP controller fixed and only searches the
8 morphology genes.  It is meant as a cheap alternative to full 592-D NSGA-II
when the controller is already good and the experiment is about body params.

Outputs are compatible with final_project_test.py:
    results/final_project_body_grid/x_best.npy
    results/final_project_body_grid/Robot.xml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from os.path import join

import gymnasium as gym
import numpy as np

import evorob.world  # registers FlatEnv-v0 / IceEnv-v0 / HillEnv-v0
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from final_project_train import DEFAULT_MLP_WARM_START, FinalWorld


ROOT_DIR = get_project_root()


TERRAINS = (
    ("flat", "FlatEnv-v0", "flat_world_file"),
    ("ice", "IceEnv-v0", "ice_world_file"),
    ("hill", "HillEnv-v0", "hill_world_file"),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    upper: float
    lower: float
    body_raw: np.ndarray


def length_to_gene(length: float) -> float:
    """Inverse of FinalWorld body mapping: L = (g + 1) / 4 + 0.1."""
    return 4.0 * (float(length) - 0.1) - 1.0


def gene_to_length(gene: np.ndarray) -> np.ndarray:
    """FinalWorld body mapping for readability in saved summaries."""
    return (np.asarray(gene, dtype=float) + 1.0) / 4.0 + 0.1


def load_checkpoint_genotype(source: str) -> np.ndarray:
    """Load x_best.npy from a checkpoint directory or direct .npy path."""
    source = os.path.expanduser(source)
    if os.path.isfile(source) and source.endswith(".npy"):
        return np.load(source, allow_pickle=True)

    if os.path.isdir(source):
        last_gen = get_last_checkpoint_dir(source)
        search_dirs = ([last_gen] if last_gen else []) + [source]
        for directory in search_dirs:
            path = join(directory, "x_best.npy")
            if os.path.isfile(path):
                return np.load(path, allow_pickle=True)

    raise FileNotFoundError(f"Could not find x_best.npy in: {source}")


def symmetric_body_raw(upper: float, lower: float) -> np.ndarray:
    """Build raw 8-gene body vector from symmetric physical lengths."""
    if not (0.1 <= upper <= 0.6 and 0.1 <= lower <= 0.6):
        raise ValueError(f"Body lengths must be in [0.1, 0.6], got {upper=}, {lower=}")
    raw = np.array(
        [
            length_to_gene(upper),
            length_to_gene(lower),
            length_to_gene(upper),
            length_to_gene(lower),
            length_to_gene(upper),
            length_to_gene(lower),
            length_to_gene(upper),
            length_to_gene(lower),
        ],
        dtype=float,
    )
    if np.any(raw < -1.0) or np.any(raw > 1.0):
        raise ValueError(f"Raw body genes out of [-1, 1]: {raw}")
    return raw


def parse_grid(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    for value in values:
        if not 0.1 <= value <= 0.6:
            raise argparse.ArgumentTypeError(
                f"grid value {value} outside body length range [0.1, 0.6]"
            )
    return values


def neutral_step_score(info: dict) -> float:
    """Same per-step neutral score used by final_project_test.py."""
    return (
        float(info.get("healthy_reward", 1.0))
        + float(info.get("x_position", 0.0))
        - float(info.get("ctrl_cost", 0.0))
        - float(info.get("cfrc_cost", 0.0))
    )


def evaluate_terrain(
    world: FinalWorld,
    env_id: str,
    world_file: str,
    *,
    n_repeats: int,
    n_steps: int,
    seed: int,
    use_neutral_reward: bool,
) -> tuple[float, float]:
    """Evaluate current world/controller on one terrain.

    Returns:
        mean score, mean episode length
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    lengths: list[int] = []

    for _ in range(n_repeats):
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=n_steps)
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))

        total = 0.0
        episode_len = 0
        done = False
        while not done:
            ctrl_obs = world.sensor_fn(obs) if world.sensor_fn is not None else obs
            action = world.controller.get_action(ctrl_obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)
            total += neutral_step_score(info) if use_neutral_reward else float(reward)
            episode_len += 1
            done = terminated or truncated

        env.close()
        scores.append(total)
        lengths.append(episode_len)

    return float(np.mean(scores)), float(np.mean(lengths))


def normalize_objectives(values: np.ndarray) -> np.ndarray:
    """Min-max normalize each terrain column for robust-score selection."""
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    out = np.zeros_like(values, dtype=float)
    non_constant = ranges > 1e-12
    out[:, non_constant] = (
        (values[:, non_constant] - mins[non_constant]) / ranges[non_constant]
    )
    return out


def maybe_write_heatmap(rows: list[dict], output_dir: str) -> None:
    """Write a small upper/lower heatmap when matplotlib is available."""
    try:
        mpl_cache = join(output_dir, ".matplotlib")
        xdg_cache = join(output_dir, ".cache")
        os.makedirs(mpl_cache, exist_ok=True)
        os.makedirs(xdg_cache, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_cache)
        os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Heatmap skipped: {exc}")
        return

    grid_rows = [r for r in rows if r["candidate"].startswith("grid_")]
    if not grid_rows:
        return

    uppers = sorted({float(r["upper"]) for r in grid_rows})
    lowers = sorted({float(r["lower"]) for r in grid_rows})
    heat = np.full((len(lowers), len(uppers)), np.nan)
    for row in grid_rows:
        x = uppers.index(float(row["upper"]))
        y = lowers.index(float(row["lower"]))
        heat[y, x] = float(row["robust_score"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat, origin="lower", aspect="auto")
    ax.set_xticks(range(len(uppers)), [f"{v:.2f}" for v in uppers])
    ax.set_yticks(range(len(lowers)), [f"{v:.2f}" for v in lowers])
    ax.set_xlabel("upper leg length [m]")
    ax.set_ylabel("lower leg length [m]")
    ax.set_title("Body-only search: normalized worst-terrain score")
    fig.colorbar(im, ax=ax, label="robust score")
    fig.tight_layout()
    fig.savefig(join(output_dir, "body_search_heatmap.png"), dpi=200)
    plt.close(fig)


def run_body_search(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    world = FinalWorld()
    warm_genotype = load_checkpoint_genotype(args.controller_checkpoint)
    if warm_genotype.size < world.n_weights:
        raise ValueError(
            f"Warm checkpoint has {warm_genotype.size} values, "
            f"but controller needs {world.n_weights}"
        )

    controller_genes = np.asarray(warm_genotype[: world.n_weights], dtype=float)
    original_body_raw = np.asarray(warm_genotype[world.n_weights : world.n_params], dtype=float)
    if original_body_raw.size != world.n_body_params:
        raise ValueError(
            f"Expected {world.n_body_params} body genes after controller, "
            f"got {original_body_raw.size}"
        )

    original_lengths = gene_to_length(original_body_raw)
    candidates: list[Candidate] = [
        Candidate(
            name="original",
            upper=float(np.mean(original_lengths[0::2])),
            lower=float(np.mean(original_lengths[1::2])),
            body_raw=original_body_raw,
        )
    ]

    for upper in args.upper_grid:
        for lower in args.lower_grid:
            candidates.append(
                Candidate(
                    name=f"grid_u{upper:.2f}_l{lower:.2f}",
                    upper=upper,
                    lower=lower,
                    body_raw=symmetric_body_raw(upper, lower),
                )
            )

    rows: list[dict] = []
    fitness_matrix = []
    genotypes = []

    print(
        f"Body-only search: {len(candidates)} candidates, "
        f"repeats={args.repeats}, steps={args.steps}, "
        f"reward={'neutral' if args.neutral_reward else 'env'}"
    )
    print(f"Frozen controller: {args.controller_checkpoint}")
    print(f"Output: {args.output_dir}\n")

    for cand_idx, candidate in enumerate(candidates):
        genotype = np.concatenate([controller_genes, candidate.body_raw])
        world.update_robot_xml(genotype)

        terrain_scores = []
        terrain_lengths = []
        for terrain_idx, (terrain_name, env_id, world_file_attr) in enumerate(TERRAINS):
            mean_score, mean_len = evaluate_terrain(
                world,
                env_id,
                getattr(world, world_file_attr),
                n_repeats=args.repeats,
                n_steps=args.steps,
                seed=args.seed + cand_idx * 1000 + terrain_idx * 100,
                use_neutral_reward=args.neutral_reward,
            )
            terrain_scores.append(mean_score)
            terrain_lengths.append(mean_len)

        fitness_matrix.append(terrain_scores)
        genotypes.append(genotype)
        rows.append(
            {
                "candidate": candidate.name,
                "upper": candidate.upper,
                "lower": candidate.lower,
                "flat": terrain_scores[0],
                "ice": terrain_scores[1],
                "hill": terrain_scores[2],
                "flat_len": terrain_lengths[0],
                "ice_len": terrain_lengths[1],
                "hill_len": terrain_lengths[2],
            }
        )
        print(
            f"{candidate.name:>18}  "
            f"u={candidate.upper:.3f} l={candidate.lower:.3f}  "
            f"flat={terrain_scores[0]:8.2f}  "
            f"ice={terrain_scores[1]:8.2f}  "
            f"hill={terrain_scores[2]:8.2f}"
        )

    fitness = np.asarray(fitness_matrix, dtype=float)
    normalized = normalize_objectives(fitness)
    robust_scores = normalized.min(axis=1)
    mean_scores = normalized.mean(axis=1)

    for idx, row in enumerate(rows):
        row["flat_norm"] = float(normalized[idx, 0])
        row["ice_norm"] = float(normalized[idx, 1])
        row["hill_norm"] = float(normalized[idx, 2])
        row["robust_score"] = float(robust_scores[idx])
        row["mean_norm_score"] = float(mean_scores[idx])

    best_idx = int(np.lexsort((-mean_scores, -robust_scores))[0])
    best_row = rows[best_idx]
    best_genotype = genotypes[best_idx]

    # Regenerate and save the matching Robot.xml for the chosen genotype.
    world.update_robot_xml(best_genotype)
    np.save(join(args.output_dir, "x_best.npy"), best_genotype)
    shutil.copy2(join(world.temp_dir.name, "Robot.xml"), join(args.output_dir, "Robot.xml"))

    csv_path = join(args.output_dir, "sweep_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "best": best_row,
        "controller_checkpoint": args.controller_checkpoint,
        "controller_params": int(world.n_weights),
        "body_params": int(world.n_body_params),
        "genotype_params": int(world.n_params),
        "reward": "neutral" if args.neutral_reward else "environment",
        "repeats": int(args.repeats),
        "steps": int(args.steps),
        "seed": int(args.seed),
    }
    with open(join(args.output_dir, "best_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    maybe_write_heatmap(rows, args.output_dir)

    print("\nBest body:")
    print(json.dumps(best_row, indent=2))
    print("\nSaved:")
    print(f"  {join(args.output_dir, 'x_best.npy')}")
    print(f"  {join(args.output_dir, 'Robot.xml')}")
    print(f"  {csv_path}")
    print("\nCompatibility check:")
    print(f"  python final_project_test.py --best_dir_path {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the trained MLP controller and grid-search body lengths."
    )
    parser.add_argument(
        "--controller-checkpoint",
        default=DEFAULT_MLP_WARM_START,
        help="Directory or .npy file containing the trained MLP/body x_best.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default=join(ROOT_DIR, "results", "final_project_body_grid"),
        help="Directory where x_best.npy, Robot.xml, and sweep_scores.csv are saved.",
    )
    parser.add_argument(
        "--upper-grid",
        type=parse_grid,
        default=parse_grid("0.20,0.30,0.40,0.50,0.60"),
        help="Comma-separated upper leg lengths in meters.",
    )
    parser.add_argument(
        "--lower-grid",
        type=parse_grid,
        default=parse_grid("0.20,0.30,0.40,0.50,0.60"),
        help="Comma-separated lower leg lengths in meters.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-reward",
        dest="neutral_reward",
        action="store_false",
        help="Use each environment's returned reward instead of final-test neutral reward.",
    )
    parser.set_defaults(neutral_reward=True)
    return parser


if __name__ == "__main__":
    run_body_search(build_parser().parse_args())
