import csv
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
BODY_PARAM_NAMES = [
    "front_left_leg",
    "front_left_ankle",
    "front_right_leg",
    "front_right_ankle",
    "back_left_leg",
    "back_left_ankle",
    "back_right_leg",
    "back_right_ankle",
]
BODY_PARAM_LABELS = {
    "front_left_leg": "Front-left upper",
    "front_left_ankle": "Front-left lower",
    "front_right_leg": "Front-right upper",
    "front_right_ankle": "Front-right lower",
    "back_left_leg": "Back-left upper",
    "back_left_ankle": "Back-left lower",
    "back_right_leg": "Back-right upper",
    "back_right_ankle": "Back-right lower",
}
MORPHOLOGY_METRIC_NAMES = [
    "avg_upper",
    "avg_lower",
    "avg_front_total",
    "avg_hind_total",
    "avg_total_limb",
]
MORPHOLOGY_METRIC_LABELS = {
    "avg_upper": "Avg. upper",
    "avg_lower": "Avg. lower",
    "avg_front_total": "Avg. front total",
    "avg_hind_total": "Avg. hind total",
    "avg_total_limb": "Avg. limb total",
}
INDIVIDUAL_DISPLAY_NAMES = {
    "specialist_obj1": "Specialist Obj. 1",
    "specialist_obj2": "Specialist Obj. 2",
    "generalist": "Generalist",
}


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
        control_weights = genotype[:self.n_weights]*0.1
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
            # multi_obj_reward = np.array([infos["z_velocity"], infos["reward_forward"]]).T # TODO
            multi_obj_reward = np.array([infos["reward_forward"]+infos["healthy_reward"], -infos["ctrl_cost"]]).T # TODO
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

    ax.set_xlabel("Fitness — Obj 1", fontsize=11)
    ax.set_ylabel("Fitness — Obj 2", fontsize=11)
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
    """Plot Pareto fronts from the raw evaluated generation, not just survivors."""
    fitness, generation_idx, source_path = _load_raw_generation_fitness(checkpoint_dir)
    if fitness is None:
        print(
            f"Could not find raw 2-objective fitness data in '{checkpoint_dir}' or its parent results directory."
        )
        return

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    save_dir = checkpoint_dir
    print(f"Plotting Pareto fronts from raw evaluated generation: {source_path}")
    plot_pareto_fronts(
        fitness,
        save_dir,
        num_generations=generation_idx,
        population_size=fitness.shape[0],
    )


def _load_raw_generation_fitness(checkpoint_dir: str):
    """Load raw evaluated-generation fitness from full_f.npy when available."""
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    checkpoint_name = os.path.basename(checkpoint_dir)
    checkpoint_parent = os.path.dirname(checkpoint_dir)
    generation_idx = int(checkpoint_name) + 1 if checkpoint_name.isdigit() else None

    candidates = []
    if checkpoint_name.isdigit():
        candidates.append(os.path.join(checkpoint_parent, "full_f.npy"))
    candidates.extend([
        os.path.join(checkpoint_dir, "full_f.npy"),
        os.path.join(checkpoint_dir, "f.npy"),
    ])

    for fitness_path in candidates:
        if not os.path.exists(fitness_path):
            continue
        try:
            loaded = np.load(fitness_path)
        except Exception as e:
            print(f"Could not load fitness data from {fitness_path}: {e}")
            continue

        if loaded.ndim == 3:
            if generation_idx is not None and 0 < generation_idx <= loaded.shape[0]:
                fitness = loaded[generation_idx - 1]
            else:
                fitness = loaded[-1]
        else:
            fitness = loaded

        if fitness.ndim == 2 and fitness.shape[1] == 2:
            return fitness, generation_idx, fitness_path

    return None, generation_idx, None


def _load_survivor_population_and_fitness(checkpoint_dir: str):
    """Load the survivor population stored in a checkpoint for evaluation."""
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    survivor_dir = last_gen if last_gen else checkpoint_dir

    population_path = os.path.join(survivor_dir, "x.npy")
    fitness_path = os.path.join(survivor_dir, "f.npy")

    population = None
    fitness = None
    if os.path.isfile(population_path):
        population = np.load(population_path, allow_pickle=True)
    if os.path.isfile(fitness_path):
        fitness = np.load(fitness_path, allow_pickle=True)
    return population, fitness, survivor_dir


def _normalise_objectives(fitness):
    """Normalise objectives to [0, 1] for balanced-selection heuristics."""
    fitness = np.asarray(fitness, dtype=float)
    mins = np.min(fitness, axis=0)
    spans = np.max(fitness, axis=0) - mins
    spans = np.where(spans > 0, spans, 1.0)
    return (fitness - mins) / spans


def _select_representative_from_sorted(pareto_indices, pareto_fitness, objective_idx, exclude=()):
    """Pick the best candidate for one objective, preferring unseen indices."""
    exclude = set(exclude)
    ranked_local = np.argsort(pareto_fitness[:, objective_idx])[::-1]
    for local_idx in ranked_local:
        global_idx = int(pareto_indices[local_idx])
        if global_idx not in exclude or len(ranked_local) <= len(exclude):
            return int(local_idx), global_idx

    local_idx = int(ranked_local[0])
    return local_idx, int(pareto_indices[local_idx])


def _select_generalist_from_pareto(pareto_indices, pareto_fitness, exclude=()):
    """Select a balanced Pareto-front solution for the generalist role."""
    exclude = set(exclude)
    normalized = _normalise_objectives(pareto_fitness)
    balance_score = np.minimum(normalized[:, 0], normalized[:, 1])
    overall_score = np.sum(normalized, axis=1)
    ranked_local = np.argsort(balance_score + 1e-6 * overall_score)[::-1]

    for local_idx in ranked_local:
        global_idx = int(pareto_indices[local_idx])
        if global_idx not in exclude or len(ranked_local) <= len(exclude):
            return int(local_idx), global_idx

    local_idx = int(ranked_local[0])
    return local_idx, int(pareto_indices[local_idx])


def select_specialists_and_generalist(population, fitness):
    """Select two specialists and one balanced generalist from the first Pareto front."""
    population = np.asarray(population)
    fitness = np.asarray(fitness, dtype=float)
    if population.ndim != 2:
        raise ValueError("Population must have shape (n_pop, n_params).")
    if fitness.ndim != 2 or fitness.shape[0] != population.shape[0] or fitness.shape[1] < 2:
        raise ValueError("Fitness must have shape (n_pop, 2) aligned with the population.")

    dummy_nsga = NSGAII(population_size=fitness.shape[0], n_opt_params=1)
    fronts, population_rank = dummy_nsga.fast_nondominated_sort(fitness)
    pareto_indices = np.asarray(fronts[0], dtype=int)
    pareto_fitness = fitness[pareto_indices]

    spec1_local, spec1_idx = _select_representative_from_sorted(
        pareto_indices,
        pareto_fitness,
        objective_idx=0,
    )
    spec2_local, spec2_idx = _select_representative_from_sorted(
        pareto_indices,
        pareto_fitness,
        objective_idx=1,
        exclude={spec1_idx},
    )
    gen_local, gen_idx = _select_generalist_from_pareto(
        pareto_indices,
        pareto_fitness,
        exclude={spec1_idx, spec2_idx},
    )

    selected = {
        "specialist_obj1": {
            "index": spec1_idx,
            "fitness": fitness[spec1_idx],
            "genotype": population[spec1_idx],
            "pareto_local_index": spec1_local,
        },
        "specialist_obj2": {
            "index": spec2_idx,
            "fitness": fitness[spec2_idx],
            "genotype": population[spec2_idx],
            "pareto_local_index": spec2_local,
        },
        "generalist": {
            "index": gen_idx,
            "fitness": fitness[gen_idx],
            "genotype": population[gen_idx],
            "pareto_local_index": gen_local,
        },
    }
    return {
        "selected": selected,
        "pareto_indices": pareto_indices,
        "population_rank": np.asarray(population_rank, dtype=int),
    }


def _extract_morphology_params(world, genotype):
    """Decode the last body genes to physical leg lengths in meters."""
    genotype = np.asarray(genotype, dtype=float)
    if genotype.size < world.n_body_params:
        raise ValueError("Genotype is shorter than the number of body parameters.")

    body_genes = genotype[-world.n_body_params:]
    body_params = (body_genes + 1.0) / 4.0 + 0.1
    return {name: float(value) for name, value in zip(BODY_PARAM_NAMES, body_params)}


def _compute_morphology_metrics(body_params):
    """Compute compact morphology summaries for specialist/generalist comparison."""
    upper_lengths = np.array([
        body_params["front_left_leg"],
        body_params["front_right_leg"],
        body_params["back_left_leg"],
        body_params["back_right_leg"],
    ])
    lower_lengths = np.array([
        body_params["front_left_ankle"],
        body_params["front_right_ankle"],
        body_params["back_left_ankle"],
        body_params["back_right_ankle"],
    ])
    front_total = np.array([
        body_params["front_left_leg"] + body_params["front_left_ankle"],
        body_params["front_right_leg"] + body_params["front_right_ankle"],
    ])
    hind_total = np.array([
        body_params["back_left_leg"] + body_params["back_left_ankle"],
        body_params["back_right_leg"] + body_params["back_right_ankle"],
    ])
    limb_total = np.array([*front_total, *hind_total])

    return {
        "avg_upper": float(np.mean(upper_lengths)),
        "avg_lower": float(np.mean(lower_lengths)),
        "avg_front_total": float(np.mean(front_total)),
        "avg_hind_total": float(np.mean(hind_total)),
        "avg_total_limb": float(np.mean(limb_total)),
    }


def write_morphology_comparison(world, selected_individuals, output_dir):
    """Write morphology table and plot for the selected Pareto-front representatives."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for label, meta in selected_individuals.items():
        body_params = _extract_morphology_params(world, meta["genotype"])
        metrics = _compute_morphology_metrics(body_params)
        row = {
            "individual": label,
            "display_name": INDIVIDUAL_DISPLAY_NAMES.get(label, label),
            "population_index": meta.get("index"),
            "objective_1": float(meta["fitness"][0]) if meta.get("fitness") is not None else np.nan,
            "objective_2": float(meta["fitness"][1]) if meta.get("fitness") is not None else np.nan,
        }
        row.update(body_params)
        row.update(metrics)
        rows.append(row)

    csv_path = os.path.join(output_dir, "morphology_comparison.csv")
    fieldnames = [
        "individual",
        "display_name",
        "population_index",
        "objective_1",
        "objective_2",
        *BODY_PARAM_NAMES,
        *MORPHOLOGY_METRIC_NAMES,
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x_body = np.arange(len(BODY_PARAM_NAMES))
    x_metrics = np.arange(len(MORPHOLOGY_METRIC_NAMES))
    width = 0.24
    offsets = np.linspace(-width, width, len(rows))
    colors = ["#B51F1F", "#007480", "#4B0082"]

    for row_idx, row in enumerate(rows):
        body_vals = [row[name] for name in BODY_PARAM_NAMES]
        metric_vals = [row[name] for name in MORPHOLOGY_METRIC_NAMES]
        label = row["display_name"]
        color = colors[row_idx % len(colors)]
        axes[0].bar(x_body + offsets[row_idx], body_vals, width=width, label=label, color=color)
        axes[1].bar(x_metrics + offsets[row_idx], metric_vals, width=width, label=label, color=color)

    axes[0].set_xticks(x_body)
    axes[0].set_xticklabels(
        [BODY_PARAM_LABELS[name] for name in BODY_PARAM_NAMES],
        rotation=30,
        ha="right",
    )
    axes[0].set_ylabel("Length [m]")
    axes[0].set_title("Morphology Parameters")
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[0].legend()

    axes[1].set_xticks(x_metrics)
    axes[1].set_xticklabels(
        [MORPHOLOGY_METRIC_LABELS[name] for name in MORPHOLOGY_METRIC_NAMES],
        rotation=15,
        ha="right",
    )
    axes[1].set_ylabel("Length [m]")
    axes[1].set_title("Derived Morphology Summaries")
    axes[1].grid(True, axis="y", alpha=0.2)
    axes[1].legend()

    fig.suptitle("Specialist vs. Generalist Morphology Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plot_path = os.path.join(output_dir, "morphology_comparison.pdf")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Morphology table saved to: {csv_path}")
    print(f"Morphology plot saved to: {plot_path}")
    return rows, csv_path, plot_path


def plot_selected_pareto_individuals(fitness, pareto_indices, selected_individuals, output_dir):
    """Highlight the chosen specialists and generalist on the Pareto front."""
    os.makedirs(output_dir, exist_ok=True)
    fitness = np.asarray(fitness, dtype=float)
    pareto_indices = np.asarray(pareto_indices, dtype=int)
    pareto_fitness = fitness[pareto_indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        fitness[:, 0],
        fitness[:, 1],
        color="lightgray",
        s=28,
        alpha=0.6,
        label="Population",
        zorder=1,
    )

    pareto_order = np.argsort(pareto_fitness[:, 0])
    pareto_sorted = pareto_fitness[pareto_order]
    ax.plot(
        pareto_sorted[:, 0],
        pareto_sorted[:, 1],
        color="#007480",
        linewidth=1.5,
        alpha=0.8,
        label="Pareto front",
        zorder=2,
    )
    ax.scatter(
        pareto_fitness[:, 0],
        pareto_fitness[:, 1],
        color="#007480",
        s=50,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    markers = {
        "specialist_obj1": "^",
        "specialist_obj2": "s",
        "generalist": "D",
    }
    colors = {
        "specialist_obj1": "#B51F1F",
        "specialist_obj2": "#D98C00",
        "generalist": "#4B0082",
    }
    for label, meta in selected_individuals.items():
        fit = np.asarray(meta["fitness"], dtype=float)
        ax.scatter(
            fit[0],
            fit[1],
            s=140,
            marker=markers.get(label, "o"),
            color=colors.get(label, "#111111"),
            edgecolors="black",
            linewidths=0.8,
            label=INDIVIDUAL_DISPLAY_NAMES.get(label, label),
            zorder=4,
        )

    ax.set_xlabel("Objective 1: reward_forward + healthy_reward")
    ax.set_ylabel("Objective 2: -ctrl_cost")
    ax.set_title("Selected Specialists and Generalist on the Pareto Front")
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.95)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "pareto_selection.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Pareto selection plot saved to: {out_path}")
    return out_path



def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,          # set to 256 for submission; lower for testing
    world: AntWorld | None = None,
    max_episode_steps: int = 1000,
    seed: int = 0,
    record_videos: bool = True,
):
    """Evaluate a Challenge-3 NSGA-II checkpoint on the hilly terrain.

    Identifies two specialists and one balanced generalist from the first
    Pareto front of the survivor population stored in the last checkpoint,
    evaluates each for n_episodes, records three videos, and writes
    comparison files.

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per individual (256 for submission).
        world:          Optional preconfigured AntWorld (useful for MLP runs).
        max_episode_steps: Episode horizon. Keep 1000 for submission artifacts.
        seed:           Base seed for reproducible evaluation.
        record_videos:  Whether to write the three rendered videos.
    """
    # --- Locate survivor checkpoint files ---
    population, fitness, survivor_dir = _load_survivor_population_and_fitness(checkpoint_dir)
    x_best_path = os.path.join(survivor_dir, "x_best.npy")
    if not os.path.isfile(x_best_path):
        x_best_path = os.path.join(os.path.abspath(checkpoint_dir), "x_best.npy")
    x_best = np.load(x_best_path, allow_pickle=True) if os.path.isfile(x_best_path) else None
    if x_best is None:
        print(f"ERROR: Could not find x_best.npy in '{checkpoint_dir}'.")
        return None

    print(f"Loaded x_best  (shape: {x_best.shape})")
    print(f"Using survivor population for evaluation: {survivor_dir}")

    # --- Identify specialist and generalist genotypes ---
    if (population is not None and fitness is not None
            and fitness.ndim == 2 and fitness.shape[1] >= 2):
        selection = select_specialists_and_generalist(population, fitness)
        for label, meta in selection["selected"].items():
            print(
                f"{INDIVIDUAL_DISPLAY_NAMES.get(label, label):<18s}: "
                f"idx={meta['index']}  f={meta['fitness']}"
            )
    else:
        print("Warning: population/fitness not found — using x_best for all three roles.")
        selection = {
            "selected": {
                "specialist_obj1": {"index": None, "fitness": None, "genotype": x_best},
                "specialist_obj2": {"index": None, "fitness": None, "genotype": x_best},
                "generalist": {"index": None, "fitness": None, "genotype": x_best},
            },
            "pareto_indices": None,
            "population_rank": None,
        }

    if world is None:
        world = AntWorld()
    controller_name = type(world.controller).__name__
    print(f"Controller: {controller_name}  |  params={world.controller.n_params}  |  genotype size={world.n_params}\n")

    individuals = {
        label: meta["genotype"]
        for label, meta in selection["selected"].items()
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
    if record_videos:
        for label, genotype in individuals.items():
            vpath = os.path.join(output_dir, f"evaluation_{video_names[label]}.mp4")
            _record_video_hill(world, genotype, max_episode_steps, seed, vpath)

    morphology_rows, morphology_csv_path, morphology_plot_path = write_morphology_comparison(
        world,
        selection["selected"],
        output_dir,
    )

    pareto_selection_path = None
    if fitness is not None and selection["pareto_indices"] is not None:
        pareto_selection_path = plot_selected_pareto_individuals(
            fitness,
            selection["pareto_indices"],
            selection["selected"],
            output_dir,
        )

    # --- Score file ---
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Challenge 3 - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller      : {controller_name} ({world.controller.n_params} params)\n")
        f.write(f"Genotype size   : {world.n_params}  (weights={world.controller.n_params}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Selection basis : survivor population ({survivor_dir})\n")
        f.write(f"Episodes/indiv. : {n_episodes}\n")
        f.write(f"Objectives      : [reward_forward+healthy_reward, -ctrl_cost]\n")
        f.write(f"Morphology CSV  : {morphology_csv_path}\n")
        f.write(f"Morphology plot : {morphology_plot_path}\n")
        if pareto_selection_path is not None:
            f.write(f"Pareto selection: {pareto_selection_path}\n")
        f.write("\n")

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

        f.write("=" * 72 + "\n")
        f.write("SELECTION ON THE FIRST PARETO FRONT\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Individual':<22s} {'Pop.Idx':>7s} {'Rank':>6s} {'Obj1':>10s} {'Obj2':>10s}\n")
        f.write("-" * 72 + "\n")
        for label, meta in selection["selected"].items():
            population_idx = meta.get("index")
            if population_idx is None:
                idx_str = "-"
                rank_str = "-"
                obj1_value = np.nan
                obj2_value = np.nan
            else:
                idx_str = str(population_idx)
                rank_str = str(int(selection["population_rank"][population_idx]) + 1)
                obj1_value = float(meta["fitness"][0])
                obj2_value = float(meta["fitness"][1])
            f.write(
                f"{label:<22s} {idx_str:>7s} {rank_str:>6s} "
                f"{obj1_value:10.2f} {obj2_value:10.2f}\n"
            )
        f.write("\n")

        f.write("=" * 72 + "\n")
        f.write("MORPHOLOGY SUMMARY [m]\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Individual':<22s} {'Avg.Up':>8s} {'Avg.Low':>8s} {'Front':>8s} {'Hind':>8s} {'Total':>8s}\n")
        f.write("-" * 72 + "\n")
        for row in morphology_rows:
            f.write(
                f"{row['individual']:<22s} {row['avg_upper']:8.3f} {row['avg_lower']:8.3f} "
                f"{row['avg_front_total']:8.3f} {row['avg_hind_total']:8.3f} {row['avg_total_limb']:8.3f}\n"
            )
        f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * 60)
    for label in individuals:
        r = results[label]
        print(f"  {label:<22s}: reward={r['reward']['mean']:.2f} +/-{r['reward']['std']:.2f}"
              f"  obj1={r['obj1']['mean']:.2f}  obj2={r['obj2']['mean']:.2f}")
    print("=" * 60)
    return results


def run_EA_single(ea_single, world, save_every=1):
    for generation_idx in trange(ea_single.n_gen):
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        should_save = save_every > 0 and (generation_idx + 1) % save_every == 0
        ea_single.tell(pop, fitnesses_gen, save_checkpoint=should_save)


def run_EA_multi(ea_multi, world, save_every=1):
    for generation_idx in trange(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        should_save = save_every > 0 and (generation_idx + 1) % save_every == 0
        ea_multi.tell(pop, fitnesses_gen, save_checkpoint=should_save)


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

    result_dir = "results/AntHill-v0/single"
    prev_best = np.load(join(get_last_checkpoint_dir(result_dir), "x_best.npy")) # load previous run
    # genotype[:-8] = prev_best # hacking a legacy

    # genotype[-8::2] = -0.6  # fix upper leg length 0.2m
    # genotype[-7::2] = 1.0     # fix lower leg length 0.6m
    # world.update_robot_xml(genotype)
    # world.visualise_individual(genotype)

    # %% Evolve open-loop mlp
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
    num_generations = 1200
    bounds = (-1, 1)

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "single")
    ea_single = EvoAlgAPI(n_parameters, population_size, num_generations, mutation_sigma, bounds, results_dir)

    # run_EA_single(ea_single, world, save_every=50)
    # plot_fitness(ea_single.full_f, results_dir)

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(results_dir, checkpoint, "x_best.npy"))
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished ES run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)
    evaluate_checkpoint(
        checkpoint_dir=results_dir,
        output_dir=join(results_dir, "evaluation"),
        n_episodes=256,
        world=world,
    )


    #%% Optimise multi-objective
    world = AntWorld()
    state_space = 27
    action_space = 8 # Change controller
    world.controller = NeuralNetworkController_Custom(input_size=state_space,
                                               output_size=action_space,
                                               hidden_size=16,
                                               load_weights=prev_best[:-action_space],
                                               )
    genotype = prev_best
    world.n_weights = world.controller.n_params
    world.n_params = world.n_weights + world.n_body_params
    n_parameters = world.n_params
    world.update_robot_xml(genotype)
    print("Number of parameters:", n_parameters)
    print("Number of weights:", world.n_weights)
    population_size = 100

    opts = {}
    opts["min"] = -1
    opts["max"] = 1
    opts["num_parents"] = population_size//2
    opts["num_generations"] = 50
    opts["mutation_prob"] = 0.3
    opts["crossover_prob"] = 0.8

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "multi-v5")
    ea_multi_obj = NSGAII(population_size,
                          n_parameters,
                          opts["num_parents"],
                          opts["num_generations"],
                          (opts["min"], opts["max"]),
                          opts["mutation_prob"],
                          opts["crossover_prob"],
                          loaded_weights=prev_best,
                          )
    ea_multi_obj.directory_name = results_dir
    # run_EA_multi(ea_multi_obj, world, save_every=20)
    # plot_fitness(ea_multi_obj.full_f, results_dir)
    # plot_pareto_fronts(
    #     ea_multi_obj.full_f,
    #     results_dir,
    #     num_generations=ea_multi_obj.n_gen,
    #     population_size=population_size,
    # )

    #%% visualise
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(checkpoint, "x_best.npy"), allow_pickle=True)
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=-1)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished NSGAII run, generating video [{video_name}]...")
    plot_pareto_fronts_from_checkpoint(checkpoint)
    world.generate_best_individual_video(env, video_name=video_name, n_steps=500)

    analysis_dir = join(checkpoint, "submission_assets")
    print(f"Generating specialist/generalist analysis in [{analysis_dir}]...")
    evaluate_checkpoint(
        checkpoint_dir=results_dir,
        output_dir=analysis_dir,
        n_episodes=256,
        world=world,
    )


if __name__ == "__main__":
    main()
