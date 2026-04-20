import os
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt

import gymnasium as gym
import imageio
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange

#TODO: set for cmaes
from evorob.algorithms.ea_api_sol import EvoAlgAPI
from evorob.algorithms.nsga_sol import NSGAII
from evorob.utils.filesys import (
    get_distinct_filename,
    get_last_checkpoint_dir,
    get_project_root,
)
from evorob.world.base import World
from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController
from evorob.world.robot.controllers.mlp import NeuralNetworkController_Custom
from evorob.world.robot.controllers.so2 import SO2Controller
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

""" 
    Morphology and Controller optimisation: Ant Hill
"""

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"


class AntWorld(World):

    def __init__(self,):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        self.controller = SO2Controller(input_size=state_space,
                                        output_size=action_space,
                                        hidden_size=action_space)

        self.n_weights = self.controller.n_params
        self.n_body_params = 8

        self.n_params = self.n_weights + self.n_body_params
        self.temp_dir = TemporaryDirectory()
        self.world_file = join(self.temp_dir.name, "AntHillEnv.xml")
        self.create_terrain_file("terrain.png")
        self.base_xml_path = join(ROOT_DIR, "evorob", "world", "robot", "assets", "hill_world.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                        [-30, 30], [-70, -30],
                        [-30, 30], [-70, -30],
                        [-30, 30], [30, 70], ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      [0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      ]

    def update_robot_xml(self, genotype: np.ndarray):
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        #% Defining the Robot environment in MuJoCo
        world = xml.parse(self.base_xml_path)
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    def create_env(self, render_mode: str = "rgb_array", n_envs: int = 1, max_episode_steps: int = 1000, reset_noise_scale=0.1, **kwargs):
        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=reset_noise_scale,
                    max_episode_steps=max_episode_steps,
                    render_mode=render_mode,
                )
                for i_env in range(n_envs)
            ]
        )
        return envs

    def geno2pheno(self, genotype):
        control_weights = genotype[:self.n_weights]
        body_params = (genotype[self.n_weights:]+1)/4+0.1
        assert len(body_params) == self.n_body_params
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle, = body_params

        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array([np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array([np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array([-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array([-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array([-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array([-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array([np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array([np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat


    def create_terrain_file(self, filename="terrain.png", width=400, depth=400):
        # 1. Create the Slope (Gradient along X)
        # 0.0 at the back, 1.0 at the front
        # TODO: Change the terrain parameters
        slope_deg = 5.0
        bump_scale = 0.1
        sigma = 3.0

        # 1. Create Linear Slope (Gradient along X)
        rise = np.tan(np.deg2rad(slope_deg))
        slope_factor = rise
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Use tan to get actual height ratio, but clip to 1.0 to stay within hfield Z-bounds
        # (Assuming max height in XML is defined as the Z-scale)
        slope_map = X * slope_factor

        # 2. Add Bumps (Noise)
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        gentle_bump = np.tanh(X*10)
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min())*gentle_bump
        noise_map = noise * bump_scale

        terrain = slope_map + noise_map
        terrain = np.clip(terrain, 0, 1)
        terrain[-1, -1] = 1
        terrain_normalized = (terrain * 255).astype(np.uint8)

        img = Image.fromarray(terrain_normalized, mode='L')
        save_path = os.path.join(self.temp_dir.name, filename)
        img.save(save_path)



    def evaluate_individual(self, genotype, n_repeats=10, n_steps=500):
        self.update_robot_xml(genotype)
        envs = self.create_env(n_envs=n_repeats, max_episode_steps=n_steps)
        self.controller.reset_controller(batch_size=n_repeats)

        rewards_full = np.zeros((n_steps, n_repeats))
        multi_obj_rewards_full = np.zeros((n_steps, n_repeats, 2))

        observations, info = envs.reset()
        done_mask = np.zeros(n_repeats, dtype=bool)
        for step in range(n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations))
            observations, rewards, dones, truncated, infos = envs.step(actions)

            # Store rewards for active environments only
            # TODO: design appropriate rewards
            rewards_full[step, ~done_mask] = rewards[~done_mask]

            # TODO: design appropriate moo-rewards
            multi_obj_reward = np.array([infos["z_velocity"], infos["reward_forward"]]).T # TODO
            # multi_obj_reward = np.array([infos["z_velocity"], -infos["ctrl_cost"]]).T # TODO
            multi_obj_rewards_full[step, ~done_mask] = multi_obj_reward[~done_mask]

            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


# ---------------------------------------------------------------------------
# Evaluation helpers (Challenge 3)
# ---------------------------------------------------------------------------

def _run_episodes_hill(world, genotype, n_episodes, max_episode_steps, seed):
    """Run n_episodes on the hill terrain, returning per-episode stats.

    Returns:
        episode_rewards: total reward per episode
        episode_obj1:    cumulative (reward_forward + healthy_reward) per episode
        episode_obj2:    cumulative (-ctrl_cost) per episode
    """
    world.update_robot_xml(genotype)
    env = gym.make(
        ENV_NAME,
        robot_path=world.world_file,
        max_episode_steps=max_episode_steps,
    )

    rng = np.random.default_rng(seed)
    episode_rewards, episode_obj1, episode_obj2 = [], [], []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        world.controller.reset_controller(batch_size=1)

        total_reward = total_obj1 = total_obj2 = 0.0
        for _ in range(max_episode_steps):
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_obj1 += float(info.get("reward_forward", 0.0)) + float(info.get("healthy_reward", 0.0))
            total_obj2 += -float(info.get("ctrl_cost", 0.0))
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_obj1.append(total_obj1)
        episode_obj2.append(total_obj2)

    env.close()
    return episode_rewards, episode_obj1, episode_obj2


def _record_video_hill(world, genotype, max_steps, seed, out_path):
    """Record one episode to out_path. Returns episode reward or None on failure."""
    try:
        world.update_robot_xml(genotype)
        env = gym.make(
            ENV_NAME,
            robot_path=world.world_file,
            max_episode_steps=max_steps,
            render_mode="rgb_array",
        )
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames, video_reward = [], 0.0

        for _ in range(max_steps):
            frames.append(env.render())
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_reward += reward
            if terminated or truncated:
                break

        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved: {out_path}")
        return video_reward
    except Exception as e:
        print(f"  Warning: video skipped ({e})")
        return None


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "values": list(values),
        "mean": float(np.mean(a)),
        "std":  float(np.std(a)),
        "best": float(np.max(a)),
        "worst": float(np.min(a)),
        "median": float(np.median(a)),
    }


def _coerce_fitness_history(full_f):
    """Return fitness history as (n_generations, n_pop, n_objectives)."""
    fitness_array = np.asarray(full_f, dtype=float)

    if fitness_array.ndim == 1:
        fitness_array = fitness_array[np.newaxis, :, np.newaxis]
    elif fitness_array.ndim == 2:
        fitness_array = fitness_array[:, :, np.newaxis]
    elif fitness_array.ndim != 3:
        raise ValueError(
            "Expected fitness history with shape (n_gen, n_pop) or (n_gen, n_pop, n_obj)."
        )

    if fitness_array.shape[0] == 0 or fitness_array.shape[1] == 0:
        raise ValueError("Cannot plot an empty fitness history.")

    return fitness_array


def plot_fitness(full_f, output_dir):
    """Save fitness-over-generations plots for one or more objectives."""
    fitness_array = _coerce_fitness_history(full_f)
    generations = np.arange(1, len(fitness_array) + 1)
    n_objectives = fitness_array.shape[2]

    fig, axes = plt.subplots(1, n_objectives, figsize=(7 * n_objectives, 5), squeeze=False)
    axes = axes.ravel()
    if n_objectives == 1:
        obj_labels = ["Fitness"]
    else:
        obj_labels = [f"Objective {i + 1}" for i in range(n_objectives)]

    for obj_idx, (ax, label) in enumerate(zip(axes, obj_labels)):
        obj_fitness = fitness_array[:, :, obj_idx]
        best_per_gen = np.max(obj_fitness, axis=1)
        mean_per_gen = np.mean(obj_fitness, axis=1)
        std_per_gen = np.std(obj_fitness, axis=1)

        ax.plot(
            generations, best_per_gen,
            label="Best", color="#B51F1F", linewidth=2, linestyle="--",
        )
        ax.plot(
            generations, mean_per_gen,
            label="Mean", color="#007480", linewidth=2,
        )
        ax.fill_between(
            generations,
            mean_per_gen - std_per_gen,
            mean_per_gen + std_per_gen,
            alpha=0.2, color="#007480", label="Mean +/- 1 std",
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fitness over Generations", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    plot_path = os.path.join(output_dir, "fitness_plot.pdf")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Fitness plot saved to: {plot_path}")


def plot_pareto_fronts(fitness, output_dir, num_generations=None, population_size=None):
    """Plot Pareto fronts for a 2-objective fitness array or history.

    Args:
        fitness:         (n_pop, 2) fitness array for one generation, or
                         (n_gen, n_pop, 2) history where the last generation is used.
        output_dir:      Directory to save the plot.
        num_generations: Number of generations (for title). Optional.
        population_size: Population size (for title). Optional.
    """
    fitness = np.asarray(fitness, dtype=float)
    if fitness.ndim == 3:
        fitness = fitness[-1]

    if fitness.ndim != 2 or fitness.shape[1] != 2:
        raise ValueError(
            "Pareto plotting expects shape (n_pop, 2) or (n_gen, n_pop, 2)."
        )
    if len(fitness) == 0:
        raise ValueError("Cannot plot a Pareto front for an empty population.")

    dummy_nsga = NSGAII(population_size=fitness.shape[0], n_opt_params=1)
    fronts, _ = dummy_nsga.fast_nondominated_sort(fitness)

    fig, ax = plt.subplots(figsize=(9, 6))
    n_fronts = len(fronts)

    # Top 3 fronts: distinct colors, connected by sorted lines
    top_colors = ["#B51F1F", "#007480", "#4B0082"]
    n_top = min(3, n_fronts)
    for i in range(n_top):
        fi = fitness[fronts[i]]
        si = np.argsort(fi[:, 0])
        fi_sorted = fi[si]
        ax.plot(
            fi_sorted[:, 0], fi_sorted[:, 1],
            color=top_colors[i], alpha=0.5, linewidth=1.2, zorder=3,
        )
        ax.scatter(
            fi[:, 0], fi[:, 1],
            label=f"Front {i + 1}",
            color=top_colors[i],
            s=50,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )

    # Remaining fronts: colormap
    if n_fronts > 3:
        remaining_cmap = plt.cm.coolwarm
        for i in range(3, n_fronts):
            fi = fitness[fronts[i]]
            t = (i - 3) / max(n_fronts - 4, 1)
            ax.scatter(
                fi[:, 0], fi[:, 1],
                label=f"Front {i + 1}" if i <= 5 else None,
                color=remaining_cmap(t),
                s=25,
                alpha=0.5,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )

    ax.set_xlabel("Fitness — Flat Terrain", fontsize=11)
    ax.set_ylabel("Fitness — Ice Terrain", fontsize=11)
    info = [f"{n_fronts} front{'s' if n_fronts > 1 else ''}"]
    if num_generations is not None:
        info.insert(0, f"gen {num_generations}")
    if population_size is not None:
        info.insert(1 if num_generations else 0, f"pop {population_size}")
    ax.set_title(f"Pareto Fronts  ({',  '.join(info)})", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pareto_path = os.path.join(output_dir, "pareto_fronts.pdf")
    fig.savefig(pareto_path, dpi=150)
    plt.close(fig)
    print(f"Pareto front plot saved to: {pareto_path}")


def plot_pareto_fronts_from_checkpoint(checkpoint_dir: str):
    """Load fitness data from a checkpoint directory or results directory and plot Pareto fronts."""
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    checkpoint_name = os.path.basename(checkpoint_dir)
    checkpoint_parent = os.path.dirname(checkpoint_dir)

    candidates = [
        os.path.join(checkpoint_dir, "f.npy"),
        os.path.join(checkpoint_dir, "full_f.npy"),
    ]
    if checkpoint_name.isdigit():
        candidates.append(os.path.join(checkpoint_parent, "full_f.npy"))

    fitness = None
    generation_idx = int(checkpoint_name) + 1 if checkpoint_name.isdigit() else None
    for fitness_path in candidates:
        if not os.path.exists(fitness_path):
            continue
        try:
            loaded = np.load(fitness_path)
        except Exception as e:
            print(f"Could not load fitness data from {fitness_path}: {e}")
            continue

        if loaded.ndim == 3 and generation_idx is not None and os.path.basename(fitness_path) == "full_f.npy":
            if 0 < generation_idx <= loaded.shape[0]:
                fitness = loaded[generation_idx - 1]
            else:
                fitness = loaded[-1]
        else:
            fitness = loaded[-1] if loaded.ndim == 3 else loaded

        if fitness.ndim == 2 and fitness.shape[1] == 2:
            break
        fitness = None

    if fitness is None:
        print(
            f"Could not find 2-objective fitness data in '{checkpoint_dir}' or its parent results directory."
        )
        return

    save_dir = checkpoint_dir
    plot_pareto_fronts(
        fitness,
        save_dir,
        num_generations=generation_idx,
        population_size=fitness.shape[0],
    )



def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,          # set to 256 for submission; lower for testing
):
    """Evaluate a Challenge-3 NSGA-II checkpoint on the hilly terrain.

    Identifies two specialists (best per objective) and a generalist (best
    combined score) from the last checkpoint's population, evaluates each for
    n_episodes, records three videos, and writes a score file.

    The controller type and genotype size are taken from the AntWorld default
    (whatever the student configured), so no hardcoded assumptions are made.
    Objectives: [reward_forward + healthy_reward,  -ctrl_cost].

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per individual (256 for submission).
    """
    max_episode_steps: int = 1000  # DO NOT CHANGE!
    seed: int = 0                  # DO NOT CHANGE!

    # --- Locate checkpoint files ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _try_load(filename):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _try_load("x_best.npy")
    if x_best is None:
        print(f"ERROR: Could not find x_best.npy in '{checkpoint_dir}'.")
        return None

    population = _try_load("x.npy")
    fitness    = _try_load("f.npy")
    print(f"Loaded x_best  (shape: {x_best.shape})")

    # --- Identify specialist and generalist genotypes ---
    if (population is not None and fitness is not None
            and fitness.ndim == 2 and fitness.shape[1] >= 2):
        spec1_idx = int(np.argmax(fitness[:, 0]))           # best forward+healthy
        spec2_idx = int(np.argmax(fitness[:, 1]))           # best efficiency
        gen_idx   = int(np.argmax(np.sum(fitness, axis=1))) # pareto-knee proxy
        spec1_g, spec2_g, gen_g = population[spec1_idx], population[spec2_idx], population[gen_idx]
        print(f"Specialist obj1 (forward): idx={spec1_idx}  f={fitness[spec1_idx]}")
        print(f"Specialist obj2 (effic.) : idx={spec2_idx}  f={fitness[spec2_idx]}")
        print(f"Generalist (best sum)    : idx={gen_idx}    f={fitness[gen_idx]}")
    else:
        print("Warning: population/fitness not found — using x_best for all three roles.")
        spec1_g = spec2_g = gen_g = x_best

    # --- AntWorld uses whatever controller the student configured ---
    world = AntWorld()
    controller_name = type(world.controller).__name__
    print(f"Controller: {controller_name}  |  params={world.controller.n_params}  |  genotype size={world.n_params}\n")

    individuals = {
        "specialist_obj1": spec1_g,
        "specialist_obj2": spec2_g,
        "generalist":      gen_g,
    }

    # --- Run episodes ---
    results = {}
    for label, genotype in individuals.items():
        print(f"Evaluating {label} ({n_episodes} episodes)...")
        rewards, obj1_vals, obj2_vals = _run_episodes_hill(
            world, genotype, n_episodes, max_episode_steps, seed
        )
        results[label] = {
            "reward": _stats(rewards),
            "obj1":   _stats(obj1_vals),
            "obj2":   _stats(obj2_vals),
        }
        r = results[label]
        print(f"  reward: {r['reward']['mean']:.2f} +/- {r['reward']['std']:.2f}  "
              f"obj1: {r['obj1']['mean']:.2f}  obj2: {r['obj2']['mean']:.2f}")

    # --- Record videos ---
    os.makedirs(output_dir, exist_ok=True)
    video_names = {
        "specialist_obj1": "specialist_forward",
        "specialist_obj2": "specialist_efficiency",
        "generalist":      "generalist",
    }
    for label, genotype in individuals.items():
        vpath = os.path.join(output_dir, f"evaluation_{video_names[label]}.mp4")
        _record_video_hill(world, genotype, max_episode_steps, seed, vpath)

    # --- Score file ---
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Challenge 3 - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller      : {controller_name} ({world.controller.n_params} params)\n")
        f.write(f"Genotype size   : {world.n_params}  (weights={world.controller.n_params}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/indiv. : {n_episodes}\n")
        f.write(f"Objectives      : [reward_forward+healthy_reward, -ctrl_cost]\n\n")

        f.write("=" * 72 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Individual':<22s} {'Rew.Mean':>9s} {'Rew.Std':>8s} {'Rew.Best':>9s} "
                f"{'Obj1.Mean':>10s} {'Obj2.Mean':>10s}\n")
        f.write("-" * 72 + "\n")
        for label in individuals:
            r = results[label]
            f.write(f"{label:<22s} {r['reward']['mean']:9.2f} {r['reward']['std']:8.2f} "
                    f"{r['reward']['best']:9.2f} {r['obj1']['mean']:10.2f} {r['obj2']['mean']:10.2f}\n")
        f.write("\n")

        for label in individuals:
            r = results[label]
            f.write("-" * 50 + "\n")
            f.write(f"{label.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, rew in enumerate(r["reward"]["values"]):
                f.write(f"  Episode {i + 1:3d}: {rew:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * 60)
    for label in individuals:
        r = results[label]
        print(f"  {label:<22s}: reward={r['reward']['mean']:.2f} +/-{r['reward']['std']:.2f}"
              f"  obj1={r['obj1']['mean']:.2f}  obj2={r['obj2']['mean']:.2f}")
    print("=" * 60)
    return results


def run_EA_single(ea_single, world):
    for _ in trange(ea_single.n_gen):
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_single.tell(pop, fitnesses_gen, save_checkpoint=True)


def run_EA_multi(ea_multi, world):
    for _ in trange(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_multi.tell(pop, fitnesses_gen, save_checkpoint=True)


def main():
    #%% Optimise single-objective
    world = AntWorld()
    n_parameters = world.n_params

    #%% Understanding the world
    genotype = np.random.uniform(-1, 1, n_parameters)
    world.update_robot_xml(genotype)
    # world.visualise_individual(genotype)
    # TODO Overwrite controller and load best run exercise 1
    state_space = 27
    action_space = 8 # Change controller
    world.controller = NeuralNetworkController_Custom(input_size=state_space,
                                               output_size=action_space,
                                               hidden_size=16)
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    genotype = np.random.uniform(-1, 1, world.n_params)

    result_dir = "prev_results/mlp"
    prev_best = np.load(join(get_last_checkpoint_dir(result_dir), "x_best.npy")) # load previous run
    genotype[:-8] = prev_best # hacking a legacy

    genotype[-8::2] = -0.6  # fix upper leg length 0.2m
    genotype[-7::2] = 1.0     # fix lower leg length 0.6m
    world.update_robot_xml(genotype)
    # world.visualise_individual(genotype)

    #%% Evolve open-loop mlp
    world = AntWorld()
    state_space = 27
    action_space = 8 # Change controller
    world.controller = NeuralNetworkController_Custom(input_size=state_space,
                                               output_size=action_space,
                                               hidden_size=16)
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    population_size = 100
    mutation_sigma = 0.3
    num_generations = 150
    bounds = (-1, 1)

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "single")
    ea_single = EvoAlgAPI(n_parameters, population_size, num_generations, mutation_sigma, bounds, results_dir)

    run_EA_single(ea_single, world)
    plot_fitness(ea_single.full_f, results_dir)

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"))
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished ES run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)


    #%% Optimise multi-objective
    world = AntWorld()
    state_space = 27
    action_space = 8 # Change controller
    world.controller = NeuralNetworkController(input_size=state_space,
                                               output_size=action_space,
                                               hidden_size=action_space)
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    print("Number of parameters:", n_parameters)
    print("Number of weights:", world.n_weights)
    population_size = 100

    opts = {}
    opts["min"] = -1
    opts["max"] = 1
    opts["num_parents"] = population_size//2
    opts["num_generations"] = 50
    opts["mutation_prob"] = 0.2
    opts["crossover_prob"] = 0.7

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "multi")
    ea_multi_obj = NSGAII(population_size,
                          n_parameters,
                          opts["num_parents"],
                          opts["num_generations"],
                          (opts["min"], opts["max"]),
                          opts["mutation_prob"],
                          opts["crossover_prob"],
                          )
    ea_multi_obj.directory_name = results_dir
    run_EA_multi(ea_multi_obj, world)
    plot_fitness(ea_multi_obj.full_f, results_dir)
    plot_pareto_fronts(
        ea_multi_obj.full_f,
        results_dir,
        num_generations=ea_multi_obj.n_gen,
        population_size=population_size,
    )

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"), allow_pickle=True)
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished NSGAII run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)


if __name__ == "__main__":
    main()
