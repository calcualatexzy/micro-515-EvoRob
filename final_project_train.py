"""
MICRO-515 Final Project — Multi-task Robot Evolution
=====================================================
Evolve a legged robot (body + controller) to walk in the +x direction across
three training environments simultaneously.  The genotype encodes both the
neural controller weights and the body morphology (leg lengths).

Training environments (3 objectives)
-------------------------------------
1  Flat  — standard ground, good friction  (FlatEnv-v0  / flat_world.xml)
2  Ice   — slippery ground, low friction   (IceEnv-v0   / ice_world.xml)
3  Hill  — procedural hilly terrain        (HillEnv-v0  / hill_world.xml)

The evaluation terrain is separate and fixed.  Students test their best
evolved robot on it using final_project_test.py — it is not trained on.
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import scipy.ndimage
from PIL import Image
from gymnasium.vector import AsyncVectorEnv

import evorob.world                         # registers EvalEnv-v0
from evorob.algorithms.nsga_sol import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_ASSETS  = join(ROOT_DIR, "evorob", "world", "robot", "assets")
MAX_EPISODE_STEPS = 1000  # fixed for leaderboard — do not change
DEFAULT_MLP_WARM_START = join(ROOT_DIR, "results/final_project_so2_climb/130")
USE_SO2_CONTROLLER = True
OBJECTIVE_LABELS = ["Flat", "Ice", "Hill"]
ARCHIVE_BODY_PATH = join(ROOT_DIR, "archive_body.npy")
ARCHIVE_FITNESS_PATH = join(ROOT_DIR, "archive_fitness.npy")


def _import_pyplot():
    """Import matplotlib in headless-safe mode and return pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _coerce_fitness_history(full_f: list | np.ndarray) -> np.ndarray:
    """Return fitness history as (n_generations, n_pop, n_objectives)."""
    fitness_array = np.asarray(full_f, dtype=float)

    if fitness_array.ndim == 1:
        fitness_array = fitness_array[np.newaxis, :, np.newaxis]
    elif fitness_array.ndim == 2:
        fitness_array = fitness_array[:, :, np.newaxis]
    elif fitness_array.ndim != 3:
        raise ValueError(
            "Expected fitness history with shape (n_gen, n_pop) or "
            "(n_gen, n_pop, n_obj)."
        )

    if fitness_array.shape[0] == 0 or fitness_array.shape[1] == 0:
        raise ValueError("Cannot plot an empty fitness history.")

    return fitness_array


def plot_fitness(full_f: list | np.ndarray, output_dir: str,
                 objective_labels: list[str] | None = None) -> str:
    """Save best/mean/std fitness-over-generations plots."""
    plt = _import_pyplot()
    fitness_array = _coerce_fitness_history(full_f)
    generations = np.arange(1, len(fitness_array) + 1)
    n_objectives = fitness_array.shape[2]

    if objective_labels is None:
        objective_labels = [f"Objective {i + 1}" for i in range(n_objectives)]
    objective_labels = objective_labels[:n_objectives]

    plot_labels = objective_labels + ["Sum"]
    fig, axes = plt.subplots(
        1, len(plot_labels), figsize=(6.5 * len(plot_labels), 5), squeeze=False
    )
    axes = axes.ravel()

    for obj_idx, (ax, label) in enumerate(zip(axes, plot_labels)):
        if obj_idx < n_objectives:
            obj_fitness = fitness_array[:, :, obj_idx]
        else:
            obj_fitness = fitness_array.sum(axis=2)

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

    plot_path = join(output_dir, "fitness_plot.pdf")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Fitness plot saved to: {plot_path}")
    return plot_path


def plot_pareto_fronts_3d(
    fitness: list | np.ndarray,
    output_dir: str,
    objective_labels: list[str] | None = None,
    num_generations: int | None = None,
    population_size: int | None = None,
) -> str:
    """Save a 3D Pareto-front visualization for the last evaluated generation."""
    plt = _import_pyplot()
    fitness_array = np.asarray(fitness, dtype=float)
    if fitness_array.ndim == 3:
        fitness_array = fitness_array[-1]

    if fitness_array.ndim != 2 or fitness_array.shape[1] != 3:
        raise ValueError(
            "3D Pareto plotting expects shape (n_pop, 3) or (n_gen, n_pop, 3)."
        )
    if len(fitness_array) == 0:
        raise ValueError("Cannot plot a Pareto front for an empty population.")

    if objective_labels is None:
        objective_labels = ["Objective 1", "Objective 2", "Objective 3"]

    dummy_nsga = NSGAII(population_size=fitness_array.shape[0], n_opt_params=1)
    fronts, _ = dummy_nsga.fast_nondominated_sort(fitness_array)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    top_colors = ["#B51F1F", "#007480", "#4B0082"]
    n_top = min(3, len(fronts))
    for i in range(n_top):
        fi = fitness_array[fronts[i]]
        ax.scatter(
            fi[:, 0], fi[:, 1], fi[:, 2],
            label=f"Front {i + 1}",
            color=top_colors[i],
            s=55,
            edgecolors="white",
            linewidths=0.5,
            depthshade=True,
        )

    if len(fronts) > 3:
        for i in range(3, len(fronts)):
            fi = fitness_array[fronts[i]]
            ax.scatter(
                fi[:, 0], fi[:, 1], fi[:, 2],
                label=f"Front {i + 1}" if i <= 5 else None,
                color="#999999",
                s=22,
                alpha=0.35,
                edgecolors="white",
                linewidths=0.2,
                depthshade=True,
            )

    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.set_zlabel(objective_labels[2])
    info = [f"{len(fronts)} front{'s' if len(fronts) > 1 else ''}"]
    if num_generations is not None:
        info.insert(0, f"gen {num_generations}")
    if population_size is not None:
        info.insert(1 if num_generations else 0, f"pop {population_size}")
    ax.set_title(f"3D Pareto Fronts  ({',  '.join(info)})")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.view_init(elev=22, azim=45)
    fig.tight_layout()

    plot_path = join(output_dir, "pareto_fronts_3d.pdf")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"3D Pareto front plot saved to: {plot_path}")
    return plot_path


def plot_training_results_from_checkpoint(results_dir: str) -> None:
    """Regenerate final-project plots from a results directory containing full_f.npy."""
    full_f_path = join(results_dir, "full_f.npy")
    if not os.path.isfile(full_f_path):
        raise FileNotFoundError(f"Could not find fitness history: {full_f_path}")
    full_f = np.load(full_f_path, allow_pickle=True)
    plot_fitness(full_f, results_dir, OBJECTIVE_LABELS)
    plot_pareto_fronts_3d(
        full_f,
        results_dir,
        OBJECTIVE_LABELS,
        num_generations=len(full_f),
        population_size=full_f.shape[1] if np.asarray(full_f).ndim >= 2 else None,
    )


def _resume_nsga_from_history(
    ea: NSGAII,
    results_dir: str,
    expected_n_params: int,
) -> int:
    """Initialize an NSGA-II instance from saved full_x/full_f history.

    Returns the next generation index to evaluate.
    """
    full_x_path = join(results_dir, "full_x.npy")
    full_f_path = join(results_dir, "full_f.npy")
    if not os.path.isfile(full_x_path) or not os.path.isfile(full_f_path):
        raise FileNotFoundError(
            f"Resume requires both {full_x_path} and {full_f_path}"
        )

    full_x = np.load(full_x_path, allow_pickle=True)
    full_f = np.load(full_f_path, allow_pickle=True)
    if full_x.ndim != 3 or full_f.ndim != 3:
        raise ValueError(
            "Resume expects full_x/full_f with shapes "
            "(n_gen, n_pop, n_params) and (n_gen, n_pop, n_obj)."
        )
    if full_x.shape[:2] != full_f.shape[:2]:
        raise ValueError(
            f"full_x/full_f population history mismatch: "
            f"{full_x.shape} vs {full_f.shape}"
        )
    if full_x.shape[2] != expected_n_params:
        raise ValueError(
            f"Checkpoint genotype size {full_x.shape[2]} does not match "
            f"current world genotype size {expected_n_params}."
        )
    if full_x.shape[1] != ea.n_pop:
        raise ValueError(
            f"Checkpoint population size {full_x.shape[1]} does not match "
            f"requested population_size {ea.n_pop}."
        )

    last_population = np.asarray(full_x[-1], dtype=float)
    last_fitness = np.asarray(full_f[-1], dtype=float)
    parents, parents_fitness = ea.sort_and_select_parents(
        last_population, last_fitness, ea.n_parents
    )
    ea.current_population = parents
    ea.fitness = parents_fitness
    ea.full_x = [np.asarray(x, dtype=float) for x in full_x]
    ea.full_f = [np.asarray(f, dtype=float) for f in full_f]
    ea.x = last_population
    ea.f = last_fitness
    ea.current_gen = len(ea.full_f)

    scalar_history = full_f.sum(axis=2)
    best_flat_idx = int(np.argmax(scalar_history))
    best_gen_idx, best_pop_idx = np.unravel_index(best_flat_idx, scalar_history.shape)
    ea.best_scalar_so_far = float(scalar_history[best_gen_idx, best_pop_idx])
    ea.x_best_so_far = np.asarray(full_x[best_gen_idx, best_pop_idx], dtype=float)
    ea.f_best_so_far = np.asarray(full_f[best_gen_idx, best_pop_idx], dtype=float)

    print(
        "Resuming NSGA-II from "
        f"{results_dir}: next_gen={ea.current_gen}, "
        f"history={full_x.shape[0]} generations, "
        f"best_gen={best_gen_idx}, best_sum={ea.best_scalar_so_far:.2f}"
    )
    return ea.current_gen


# ---------------------------------------------------------------------------
# FinalWorld — body + brain co-evolution across multiple terrains
# ---------------------------------------------------------------------------

class FinalWorld(World):
    """Translates a genotype into a robot phenotype and evaluates it.

    The genotype is a 1-D array: [controller_params | body_params].
    Each call to evaluate_individual generates the robot body XML, injects it
    into every terrain template, then runs the controller in parallel episodes.
    """

    def __init__(self):
        # Choose your controller — swap for your own MLP, SO2Controller, Hebbian, or custom.
        # Whatever you choose determines self.n_weights (controller parameter count).
        #
        if USE_SO2_CONTROLLER:
            from evorob.world.robot.controllers.so2 import SO2Controller
            self.controller = SO2Controller(input_size=27, output_size=8, hidden_size=8)
        else:
            from evorob.world.robot.controllers.mlp import NeuralNetworkController
            self.controller = NeuralNetworkController(
                input_size=27, output_size=8, hidden_size=16
            )

        self.n_weights     = self.controller.n_params
        self.controller_param_scale = 0.1
        # Independent morphology genes for all 8 ant leg segments:
        # [front left upper, front left lower, front right upper, front right lower,
        #  back left upper, back left lower, back right upper, back right lower].
        self.n_body_params = 8
        self.n_params      = self.n_weights + self.n_body_params

        # Temporary directory holds AntRobot.xml + one combined world XML per terrain
        self.temp_dir        = TemporaryDirectory()
        self.flat_world_file = join(self.temp_dir.name, "WorldFlat.xml")
        self.ice_world_file  = join(self.temp_dir.name, "WorldIce.xml")
        self.hill_world_file = join(self.temp_dir.name, "WorldHill.xml")
        self.world_file      = self.hill_world_file  # default for visualisation

        # Joint geometry — matches the AntRobot topology
        self.joint_limits = [
            [-30, 30], [30, 70],
            [-30, 30], [-70, -30],
            [-30, 30], [-70, -30],
            [-30, 30], [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
        ]

        # Custom sensor function — intercepts the raw env observation before it
        # reaches the controller.  Set to any callable obs -> obs' to filter,
        # augment, or reshape observations.  The controller input_size must match
        # the output of this function.
        #
        # Example — use only joint angles and velocities (14 values):
        #   self.sensor_fn = lambda obs: obs[:14]
        #   self.controller = NeuralNetworkController(input_size=14, ...)
        self.sensor_fn = None

        self._create_terrain_file("terrain.png")

    # ------------------------------------------------------------------
    # Genotype → phenotype
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray):
        """Decode genotype into controller weights and body parameters.

        Splits genotype into:
          genotype[:n_weights]  → controller
          genotype[n_weights:]  → 8 independent body params via (g+1)/4 + 0.1

        Returns (points, connectivity_mat) for AntRobot construction.
        """
        control_params = genotype[:self.n_weights] * self.controller_param_scale
        body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1
        self.controller.geno2pheno(control_params)

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

    # ------------------------------------------------------------------
    # Robot XML generation
    # ------------------------------------------------------------------

    def update_robot_xml(self, genotype: np.ndarray) -> None:
        """Build robot body XML from genotype and inject into every terrain template.

        Writes AntRobot.xml to temp_dir, then creates one combined world XML per
        terrain (flat, ice, hill) by appending an <include> to the template.
        """
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(
            points, connectivity_mat, self.joint_limits, self.joint_axis,
            name="Robot", verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)          # → Robot.xml

        for template, world_file in [
            (join(_ASSETS, "flat_world.xml"), self.flat_world_file),
            (join(_ASSETS, "ice_world.xml"),  self.ice_world_file),
            (join(_ASSETS, "hill_world.xml"), self.hill_world_file),
        ]:
            tree = xml.parse(template)
            root = tree.getroot()
            root.append(xml.Element("include", attrib={"file": "Robot.xml"}))
            with open(world_file, "w") as f:
                f.write(xml.tostring(root, encoding="unicode"))

    def _create_terrain_file(self, filename: str, width: int = 200, depth: int = 400):
        """Hill terrain PNG: smooth start, bumpy middle, smooth end."""
        slope_deg = 5.0
        bump_scale = 0.08
        sigma = 4.0

        rise = np.tan(np.deg2rad(slope_deg))
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Smooth slope: starts flat, gradually rises
        slope_map = np.clip(X * rise, 0, 1)

        # Bell-shaped envelope: smooth at both ends, bumpy in the middle
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        bump_envelope = np.sin(np.pi * X)  # 0 at start, peaks at mid, 0 at end
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * bump_envelope
        noise_map = noise * bump_scale

        terrain = np.clip(slope_map + noise_map, 0, 1)
        terrain[-1, -1] = 1  # ensure max value for normalization

        img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")
        img.save(join(self.temp_dir.name, filename))

    # ------------------------------------------------------------------
    # Per-terrain evaluation
    # ------------------------------------------------------------------

    def _run_env(self, env_id: str, world_file: str, n_repeats: int, n_steps: int) -> float:
        """Run n_repeats parallel episodes and return the mean total reward."""
        envs = AsyncVectorEnv([
            (lambda eid, wf: lambda: gym.make(
                eid, robot_path=wf, max_episode_steps=n_steps
            ))(env_id, world_file)
            for _ in range(n_repeats)
        ])
        self.controller.reset_controller(batch_size=n_repeats)
        rewards = np.zeros((n_steps, n_repeats))
        obs, _ = envs.reset()
        if self.sensor_fn is not None:
            obs = self.sensor_fn(obs)
        done = np.zeros(n_repeats, dtype=bool)
        for t in range(n_steps):
            actions = np.where(done[:, None], 0, self.controller.get_action(obs))
            obs, r, terminated, truncated, _ = envs.step(actions)
            if self.sensor_fn is not None:
                obs = self.sensor_fn(obs)
            rewards[t, ~done] = r[~done]
            done |= terminated | truncated
            if done.all():
                break
        envs.close()
        return float(rewards.sum(axis=0).mean())

    def _eval_flat(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("FlatEnv-v0", self.flat_world_file, n_repeats, n_steps)

    def _eval_ice(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("IceEnv-v0", self.ice_world_file, n_repeats, n_steps)

    def _eval_hill(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("HillEnv-v0", self.hill_world_file, n_repeats, n_steps)

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a HillEnv-v0 instance (used for visualisation)."""
        return gym.make("HillEnv-v0", robot_path=self.hill_world_file,
                        render_mode=render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Combined fitness for NSGA-II
    # ------------------------------------------------------------------

    def evaluate_individual(self, genotype: np.ndarray,
                            n_repeats: int = 4, n_steps: int = 500) -> np.ndarray:
        """Evaluate one genotype on all three training environments.

        Returns a 1-D array of three objective values: [flat, ice, hill].
        """
        self.update_robot_xml(genotype)
        return np.array([
            self._eval_flat(n_repeats, n_steps),
            self._eval_ice(n_repeats, n_steps),
            self._eval_hill(n_repeats, n_steps),
        ])


class MLPBody4SO2ControlWorld(FinalWorld):
    """Final-project world that freezes a Challenge3 MLP body and evolves SO2 only."""

    def __init__(self, fixed_body_raw: np.ndarray, controller_param_scale: float = 1.0):
        super().__init__()
        fixed_body_raw = np.asarray(fixed_body_raw, dtype=float)
        if fixed_body_raw.size != self.n_body_params:
            raise ValueError(
                f"Expected {self.n_body_params} fixed body genes, got "
                f"{fixed_body_raw.size}."
            )
        self.fixed_body_raw = fixed_body_raw.copy()
        self.controller_param_scale = float(controller_param_scale)
        self.n_controller_params = self.n_weights
        self.n_full_params = self.n_weights + self.n_body_params
        # The EA in MLPbody4SO2control mode optimizes only SO2 controller genes.
        self.n_params = self.n_controller_params

    def as_full_genotype(self, controller_genotype: np.ndarray) -> np.ndarray:
        controller_genotype = np.asarray(controller_genotype, dtype=float)
        if controller_genotype.size != self.n_controller_params:
            raise ValueError(
                f"Expected {self.n_controller_params} SO2 controller genes, "
                f"got {controller_genotype.size}."
            )
        return np.concatenate([controller_genotype, self.fixed_body_raw])

    def geno2pheno(self, genotype: np.ndarray):
        genotype = np.asarray(genotype, dtype=float)
        if genotype.size == self.n_controller_params:
            genotype = self.as_full_genotype(genotype)
        elif genotype.size != self.n_full_params:
            raise ValueError(
                f"Expected either {self.n_controller_params} controller genes "
                f"or {self.n_full_params} full genes, got {genotype.size}."
            )
        return super().geno2pheno(genotype)

# ---------------------------------------------------------------------------
# Warm-start helpers
# ---------------------------------------------------------------------------

def _numeric_parent_key(path: str) -> tuple[int, str]:
    """Sort checkpoint files by numeric parent directory, newest first."""
    parent = os.path.basename(os.path.dirname(path.rstrip("/")))
    return (int(parent) if parent.isdigit() else -1, path)


def _load_checkpoint_genotype(source: str) -> np.ndarray:
    """Load x_best.npy from a directory, a .npy file, or a zipped results folder."""
    if source is None:
        raise ValueError("No warm-start source provided.")

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
        raise FileNotFoundError(f"x_best.npy not found in warm-start directory: {source}")


def _warm_start_population_with_mlp(
    population: np.ndarray,
    mlp_params: np.ndarray,
    bounds: tuple[float, float],
    noise_scale: float,
) -> np.ndarray:
    """Set generation-0 MLP genes from a pretrained controller.

    The first individual receives the pretrained MLP exactly.  The remaining
    individuals receive small bounded perturbations so NSGA-II can still evolve
    the controller slice (differential mutation needs initial diversity).
    Body parameters stay as sampled by the EA.
    """
    warm_population = population.copy()
    n_weights = mlp_params.size
    warm_population[0, :n_weights] = mlp_params

    if len(warm_population) > 1:
        noise = np.random.normal(
            loc=0.0,
            scale=noise_scale,
            size=(len(warm_population) - 1, n_weights),
        )
        warm_population[1:, :n_weights] = mlp_params + noise

    warm_population[:, :n_weights] = np.clip(
        warm_population[:, :n_weights], bounds[0], bounds[1]
    )
    return warm_population


def load_archive_best_body(
    archive_body_path: str = ARCHIVE_BODY_PATH,
    archive_fitness_path: str = ARCHIVE_FITNESS_PATH,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Load the archive body with the highest archived fitness."""
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

    best_idx = int(np.argmax(selection_values))
    best_body = np.asarray(archive_body[best_idx], dtype=float)
    best_fitness = float(selection_values[best_idx])
    return best_idx, best_fitness, best_body, fitness


# ---------------------------------------------------------------------------
# Neutral leaderboard evaluation  (TA-graded — do not modify)
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,          # set to 256 for submission; lower for testing
) -> dict | None:
    """Evaluate the best genotype from a checkpoint on all three training terrains.

    Loads x_best.npy, evaluates it on flat, ice, and hill for n_episodes each,
    prints per-episode scores, records one video per terrain, and writes a score file.

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per terrain (256 for submission).
    """
    MAX_STEPS = MAX_EPISODE_STEPS   # DO NOT CHANGE
    SEED      = 0                   # DO NOT CHANGE

    # --- Locate checkpoint ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _load(fname):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _load("x_best.npy")
    if x_best is None:
        print(f"ERROR: x_best.npy not found in '{checkpoint_dir}'.")
        return None
    print(f"Loaded x_best  (shape: {x_best.shape})")

    world = FinalWorld()
    world.update_robot_xml(x_best)
    ctrl_name = type(world.controller).__name__
    print(f"Controller: {ctrl_name}  |  n_weights={world.n_weights}"
          f"  |  genotype size={world.n_params}\n")

    terrains = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    def _neutral(info: dict) -> float:
        return (float(info.get("healthy_reward", 1.0))
                + float(info.get("x_position",   0.0))
                - float(info.get("ctrl_cost",     0.0))
                - float(info.get("cfrc_cost",     0.0)))

    def _stats(values: list) -> dict:
        arr = np.asarray(values)
        return dict(mean=float(arr.mean()), std=float(arr.std()),
                    best=float(arr.max()), worst=float(arr.min()), values=values)

    def _run(env_id: str, world_file: str) -> list:
        rng = np.random.default_rng(SEED)
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=MAX_STEPS)
        rewards = []
        for ep in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += _neutral(info)
                done = terminated or truncated
            rewards.append(total)
        env.close()
        return rewards

    def _record(env_id: str, world_file: str, out_path: str) -> None:
        try:
            import imageio
            env = gym.make(env_id, robot_path=world_file,
                           render_mode="rgb_array", max_episode_steps=MAX_STEPS)
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=SEED)
            frames = []
            for _ in range(MAX_STEPS):
                frames.append(env.render())
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            env.close()
            imageio.mimwrite(out_path, frames, fps=20)
            print(f"  Video: {out_path}")
        except Exception as exc:
            print(f"  Video skipped: {exc}")

    # --- Evaluate on each terrain ---
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for terrain_name, (env_id, world_file) in terrains.items():
        print(f"  Running {terrain_name}  ({n_episodes} episodes)...", flush=True)
        results[terrain_name] = _stats(_run(env_id, world_file))

    # Per-episode 3-column table
    t_names = list(results.keys())
    col_w = 12
    hdr = f"  {'Ep':>4}   " + "   ".join(f"{n.capitalize():>{col_w}}" for n in t_names)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for ep in range(n_episodes):
        row = f"  {ep + 1:>4}   " + "   ".join(
            f"{results[n]['values'][ep]:>{col_w}.2f}" for n in t_names
        )
        print(row)
    print(sep)
    print(f"  {'mean':>4}   " + "   ".join(
        f"{results[n]['mean']:>{col_w}.2f}" for n in t_names
    ))
    print(f"  {'std':>4}   " + "   ".join(
        f"{results[n]['std']:>{col_w}.2f}" for n in t_names
    ))
    print()

    # --- Record one video per terrain ---
    print("Recording videos...")
    for terrain_name, (env_id, world_file) in terrains.items():
        _record(env_id, world_file, join(output_dir, f"evaluation_{terrain_name}.mp4"))

    # --- Score file ---
    score_path = join(output_dir, "evaluation_score.txt")
    col = 60
    with open(score_path, "w") as f:
        f.write("=" * col + "\n")
        f.write("MICRO-515 Final Project — Evaluation Results\n")
        f.write("=" * col + "\n\n")
        f.write(f"Controller      : {ctrl_name} ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/terrain: {n_episodes}\n")
        f.write(f"Reward          : healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n")

        f.write("=" * col + "\n")
        f.write("SUMMARY\n")
        f.write("=" * col + "\n")
        f.write(f"{'Terrain':<8} {'Mean':>9} {'Std':>8} {'Best':>9} {'Worst':>9}\n")
        f.write("-" * col + "\n")
        for terrain_name, r in results.items():
            f.write(f"{terrain_name:<8} {r['mean']:9.2f} {r['std']:8.2f}"
                    f" {r['best']:9.2f} {r['worst']:9.2f}\n")
        f.write("\n")

        for terrain_name, r in results.items():
            f.write("-" * 50 + "\n")
            f.write(f"{terrain_name.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, v in enumerate(r["values"]):
                f.write(f"  Episode {i + 1:3d}: {v:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * col)
    for terrain_name, r in results.items():
        print(f"  {terrain_name:<6}: {r['mean']:8.2f} ± {r['std']:7.2f}"
              f"  best={r['best']:.2f}  worst={r['worst']:.2f}")
    print("=" * col)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_multi_task_evolution(
    num_generations: int = 70,
    population_size: int = 250,
    n_parents:       int = 250,
    n_repeats:       int = 4,
    n_steps:         int = 500,
    mutation_prob:   float = 0.3,
    crossover_prob:  float = 0.9,
    bounds:          tuple = (-1, 1),
    ckpt_interval:   int = 10,
    results_dir:     str = None,
    random_seed:     int = 42,
    mlp_warm_start_source: str | None = None,
    mlp_warm_start_noise: float = 0.05,
    resume:          bool = False,
) -> None:
    np.random.seed(random_seed)

    world = FinalWorld()
    print(f"Genotype : {world.n_params} params"
          f"  (controller={world.n_weights}, body={world.n_body_params})")

    warm_start_mlp = None
    if mlp_warm_start_source is not None:
        warm_start_genotype = _load_checkpoint_genotype(mlp_warm_start_source)
        world.visualise_individual(warm_start_genotype, n_steps=n_steps)
        warm_start_mlp = np.asarray(warm_start_genotype[:world.n_weights], dtype=float)
        if warm_start_mlp.size != world.n_weights:
            raise ValueError(
                "Warm-start checkpoint does not contain enough MLP parameters: "
                f"need {world.n_weights}, got {warm_start_genotype.size} total values."
            )
        print(
            "Warm-start MLP: "
            f"{mlp_warm_start_source} ({world.n_weights} controller params)"
        )

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "final_project")

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    n_obj = 3
    start_gen = 0
    if resume:
        start_gen = _resume_nsga_from_history(ea, results_dir, world.n_params)

    total_target_generations = start_gen + num_generations
    print(f"\nRunning {num_generations} additional generations  pop={population_size}")
    print(f"Objectives : [flat, ice, hill]")
    print(f"Checkpoints: {results_dir}\n")

    os.makedirs(results_dir, exist_ok=True)
    _best_xml_stage = join(results_dir, "_best_robot.xml")  # staging copy of best robot
    if ea.x_best_so_far is not None:
        world.update_robot_xml(ea.x_best_so_far)
        shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)
        _best_scalar = float(ea.best_scalar_so_far)
    else:
        _best_scalar = -np.inf

    for gen in range(start_gen, total_target_generations):
        pop = ea.ask()
        if gen == 0 and warm_start_mlp is not None:
            pop = _warm_start_population_with_mlp(
                pop, warm_start_mlp, bounds=bounds, noise_scale=mlp_warm_start_noise
            )
        fitnesses = np.empty((len(pop), n_obj))
        for idx, genotype in enumerate(pop):
            fitnesses[idx] = world.evaluate_individual(
                genotype, n_repeats=n_repeats, n_steps=n_steps
            )
            scalar = float(fitnesses[idx].sum())
            if scalar > _best_scalar:
                _best_scalar = scalar
                shutil.copy2(
                    join(world.temp_dir.name, "Robot.xml"),
                    _best_xml_stage,
                )
        save_ckpt = (gen % ckpt_interval == 0) or (gen == total_target_generations - 1)
        ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)
        if save_ckpt:
            shutil.copy2(
                _best_xml_stage,
                join(results_dir, str(gen), "Robot.xml"),
            )

    # --- Training summary ---
    best_f = ea.f_best_so_far  # shape (3,) for NSGA-II
    if ea.x_best_so_far is not None:
        world.update_robot_xml(ea.x_best_so_far)
        np.save(join(results_dir, "x_best.npy"), np.asarray(ea.x_best_so_far))
        np.save(join(results_dir, "f_best.npy"), np.asarray(ea.f_best_so_far))
        shutil.copy2(join(world.temp_dir.name, "Robot.xml"), join(results_dir, "Robot.xml"))

    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Resume          : {resume}\n")
        f.write(f"Start generation: {start_gen}\n")
        f.write(f"Added generations: {num_generations}\n")
        f.write(f"Total generations: {len(ea.full_f)}\n")
        f.write(f"Population size : {population_size}\n")
        f.write(f"Controller      : {type(world.controller).__name__}"
                f"  ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n\n")
        f.write("Best individual (highest sum of objectives):\n")
        labels = ["flat", "ice", "hill"]
        for label, val in zip(labels, best_f):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'sum':<6}: {float(best_f.sum()):10.2f}\n")
    print(f"\nTraining summary saved to: {score_path}")

    try:
        plot_fitness(ea.full_f, results_dir, OBJECTIVE_LABELS)
        plot_pareto_fronts_3d(
            ea.full_f,
            results_dir,
            OBJECTIVE_LABELS,
            num_generations=len(ea.full_f),
            population_size=population_size,
        )
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


def load_mlp_body_from_checkpoint(
    checkpoint_source: str,
    *,
    mlp_hidden_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load the 8 body genes from a Challenge3 MLP checkpoint.

    Returns:
        full checkpoint genotype, body gene slice, resolved x_best.npy path.
    """
    genotype = _load_checkpoint_genotype(checkpoint_source)
    from evorob.world.robot.controllers.mlp import NeuralNetworkController

    mlp_controller = NeuralNetworkController(
        input_size=27,
        output_size=8,
        hidden_size=mlp_hidden_size,
    )
    expected_size = mlp_controller.n_params + 8
    if genotype.size < expected_size:
        raise ValueError(
            f"MLP body checkpoint has {genotype.size} genes, but hidden_size="
            f"{mlp_hidden_size} expects at least {expected_size} "
            f"({mlp_controller.n_params} controller + 8 body)."
        )

    body_raw = np.asarray(
        genotype[mlp_controller.n_params : mlp_controller.n_params + 8],
        dtype=float,
    )
    return genotype, body_raw, os.path.abspath(os.path.expanduser(checkpoint_source))


def run_mlp_body4so2control(
    num_generations: int = 70,
    population_size: int = 250,
    n_parents: int = 250,
    n_repeats: int = 4,
    n_steps: int = 500,
    mutation_prob: float = 0.3,
    crossover_prob: float = 0.9,
    bounds: tuple = (-1, 1),
    ckpt_interval: int = 10,
    results_dir: str | None = None,
    random_seed: int = 42,
    mlp_body_checkpoint: str | None = None,
    mlp_body_hidden_size: int = 16,
    so2_controller_param_scale: float = 1.0,
) -> None:
    """Evolve only the SO2 controller while freezing a body from an MLP checkpoint.

    The saved root ``x_best.npy`` is a full final-project genotype:
    ``[SO2 controller genes | fixed MLP body genes]``.
    """
    np.random.seed(random_seed)

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "MLPbody4SO2control")
    if mlp_body_checkpoint is None:
        mlp_body_checkpoint = join(ROOT_DIR, "results", "AntHill-v0", "single")

    _, fixed_body_raw, resolved_body_source = load_mlp_body_from_checkpoint(
        mlp_body_checkpoint,
        mlp_hidden_size=mlp_body_hidden_size,
    )

    world = MLPBody4SO2ControlWorld(
        fixed_body_raw=fixed_body_raw,
        controller_param_scale=so2_controller_param_scale,
    )
    print("Mode: MLPbody4SO2control")
    print(f"Fixed MLP body source : {resolved_body_source}")
    print(f"Fixed body raw genes  : {fixed_body_raw}")
    print(f"Fixed body lengths [m]: {(fixed_body_raw + 1.0) / 4.0 + 0.1}")
    print(
        f"Optimizing SO2 controller only: {world.n_controller_params} params; "
        f"saved full genotype: {world.n_full_params} params"
    )
    print(f"SO2 controller param scale during training: {world.controller_param_scale}")

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    os.makedirs(results_dir, exist_ok=True)
    full_x_history: list[np.ndarray] = []
    best_full_so_far: np.ndarray | None = None
    best_scalar_so_far = -np.inf
    _best_xml_stage = join(results_dir, "_best_robot.xml")

    for gen in range(num_generations):
        controller_population = ea.ask()
        full_population = np.asarray(
            [world.as_full_genotype(ind) for ind in controller_population],
            dtype=float,
        )
        fitnesses = np.empty((len(controller_population), 3), dtype=float)

        for idx, controller_genotype in enumerate(controller_population):
            fitnesses[idx] = world.evaluate_individual(
                controller_genotype,
                n_repeats=n_repeats,
                n_steps=n_steps,
            )
            scalar = float(fitnesses[idx].sum())
            if scalar > best_scalar_so_far:
                best_scalar_so_far = scalar
                best_full_so_far = full_population[idx].copy()
                shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)

        full_x_history.append(full_population)
        save_ckpt = (gen % ckpt_interval == 0) or (gen == num_generations - 1)
        ea.tell(controller_population, fitnesses, save_checkpoint=save_ckpt)

        if save_ckpt:
            ckpt_dir = join(results_dir, str(gen))
            os.makedirs(ckpt_dir, exist_ok=True)
            if ea.x_best_so_far is not None:
                np.save(join(ckpt_dir, "x_controller_best.npy"), ea.x_best_so_far)
                np.save(join(ckpt_dir, "x_best.npy"), world.as_full_genotype(ea.x_best_so_far))
            np.save(join(ckpt_dir, "x_full.npy"), full_population)
            shutil.copy2(_best_xml_stage, join(ckpt_dir, "Robot.xml"))

    if ea.x_best_so_far is None or best_full_so_far is None:
        raise RuntimeError("No best individual was recorded during MLPbody4SO2control.")

    best_controller = np.asarray(ea.x_best_so_far, dtype=float)
    best_full = world.as_full_genotype(best_controller)
    world.update_robot_xml(best_controller)

    np.save(join(results_dir, "x_best.npy"), best_full)
    np.save(join(results_dir, "x_controller_best.npy"), best_controller)
    np.save(join(results_dir, "f_best.npy"), np.asarray(ea.f_best_so_far))
    np.save(join(results_dir, "full_x.npy"), np.asarray(full_x_history))
    np.save(join(results_dir, "full_x_controller.npy"), np.asarray(ea.full_x))
    np.save(join(results_dir, "full_f.npy"), np.asarray(ea.full_f))
    shutil.copy2(join(world.temp_dir.name, "Robot.xml"), join(results_dir, "Robot.xml"))

    metadata = {
        "mode": "MLPbody4SO2control",
        "mlp_body_checkpoint": resolved_body_source,
        "mlp_body_hidden_size": int(mlp_body_hidden_size),
        "fixed_body_raw": fixed_body_raw.tolist(),
        "fixed_body_lengths_m": ((fixed_body_raw + 1.0) / 4.0 + 0.1).tolist(),
        "controller": "SO2Controller",
        "controller_params": int(world.n_controller_params),
        "body_params": int(world.n_body_params),
        "full_genotype_params": int(world.n_full_params),
        "optimized_params": int(world.n_params),
        "so2_controller_param_scale": float(world.controller_param_scale),
        "num_generations": int(num_generations),
        "population_size": int(population_size),
        "n_parents": int(n_parents),
        "n_repeats": int(n_repeats),
        "n_steps": int(n_steps),
        "random_seed": int(random_seed),
        "best_fitness": np.asarray(ea.f_best_so_far, dtype=float).tolist(),
        "best_scalar": float(np.asarray(ea.f_best_so_far, dtype=float).sum()),
    }
    with open(join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — MLPbody4SO2control Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fixed body source : {resolved_body_source}\n")
        f.write(f"SO2 scale         : {world.controller_param_scale}\n")
        f.write(f"Optimized params  : {world.n_params}\n")
        f.write(f"Saved full params : {world.n_full_params}\n\n")
        labels = ["flat", "ice", "hill"]
        for label, val in zip(labels, np.asarray(ea.f_best_so_far, dtype=float)):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'sum':<6}: {float(np.asarray(ea.f_best_so_far).sum()):10.2f}\n")
    print(f"\nMLPbody4SO2control summary saved to: {score_path}")

    try:
        plot_fitness(ea.full_f, results_dir, OBJECTIVE_LABELS)
        plot_pareto_fronts_3d(
            ea.full_f,
            results_dir,
            OBJECTIVE_LABELS,
            num_generations=len(ea.full_f),
            population_size=population_size,
        )
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


def run_archive_body4so2control(
    num_generations: int = 70,
    population_size: int = 250,
    n_parents: int = 250,
    n_repeats: int = 4,
    n_steps: int = 500,
    mutation_prob: float = 0.3,
    crossover_prob: float = 0.9,
    bounds: tuple = (-1, 1),
    ckpt_interval: int = 10,
    results_dir: str | None = None,
    random_seed: int = 42,
    archive_body_path: str = ARCHIVE_BODY_PATH,
    archive_fitness_path: str = ARCHIVE_FITNESS_PATH,
    so2_controller_param_scale: float = 1.0,
) -> None:
    """Evolve only the SO2 controller while freezing the archive's best body.

    The saved root ``x_best.npy`` is a full final-project genotype:
    ``[SO2 controller genes | archive_body[argmax(archive_fitness)]]``.
    """
    np.random.seed(random_seed)

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "Archivebody4SO2control")

    best_idx, best_archive_fitness, fixed_body_raw, archive_fitness = load_archive_best_body(
        archive_body_path,
        archive_fitness_path,
    )

    world = MLPBody4SO2ControlWorld(
        fixed_body_raw=fixed_body_raw,
        controller_param_scale=so2_controller_param_scale,
    )
    print("Mode: Archivebody4SO2control")
    print(f"Archive body source  : {os.path.abspath(archive_body_path)}")
    print(f"Archive fitness source: {os.path.abspath(archive_fitness_path)}")
    print(f"Archive best index   : {best_idx}")
    print(f"Archive best fitness : {best_archive_fitness:.6f}")
    print(f"Fixed body raw genes : {fixed_body_raw}")
    print(f"Fixed body lengths [m]: {(fixed_body_raw + 1.0) / 4.0 + 0.1}")
    print(
        f"Optimizing SO2 controller only: {world.n_controller_params} params; "
        f"saved full genotype: {world.n_full_params} params"
    )
    print(f"SO2 controller param scale during training: {world.controller_param_scale}")

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    os.makedirs(results_dir, exist_ok=True)
    full_x_history: list[np.ndarray] = []
    best_full_so_far: np.ndarray | None = None
    best_scalar_so_far = -np.inf
    _best_xml_stage = join(results_dir, "_best_robot.xml")

    for gen in range(num_generations):
        controller_population = ea.ask()
        full_population = np.asarray(
            [world.as_full_genotype(ind) for ind in controller_population],
            dtype=float,
        )
        fitnesses = np.empty((len(controller_population), 3), dtype=float)

        for idx, controller_genotype in enumerate(controller_population):
            fitnesses[idx] = world.evaluate_individual(
                controller_genotype,
                n_repeats=n_repeats,
                n_steps=n_steps,
            )
            scalar = float(fitnesses[idx].sum())
            if scalar > best_scalar_so_far:
                best_scalar_so_far = scalar
                best_full_so_far = full_population[idx].copy()
                shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)

        full_x_history.append(full_population)
        save_ckpt = (gen % ckpt_interval == 0) or (gen == num_generations - 1)
        ea.tell(controller_population, fitnesses, save_checkpoint=save_ckpt)

        if save_ckpt:
            ckpt_dir = join(results_dir, str(gen))
            os.makedirs(ckpt_dir, exist_ok=True)
            if ea.x_best_so_far is not None:
                np.save(join(ckpt_dir, "x_controller_best.npy"), ea.x_best_so_far)
                np.save(join(ckpt_dir, "x_best.npy"), world.as_full_genotype(ea.x_best_so_far))
            np.save(join(ckpt_dir, "x_full.npy"), full_population)
            shutil.copy2(_best_xml_stage, join(ckpt_dir, "Robot.xml"))

    if ea.x_best_so_far is None or best_full_so_far is None:
        raise RuntimeError("No best individual was recorded during Archivebody4SO2control.")

    best_controller = np.asarray(ea.x_best_so_far, dtype=float)
    best_full = world.as_full_genotype(best_controller)
    world.update_robot_xml(best_controller)

    np.save(join(results_dir, "x_best.npy"), best_full)
    np.save(join(results_dir, "x_controller_best.npy"), best_controller)
    np.save(join(results_dir, "f_best.npy"), np.asarray(ea.f_best_so_far))
    np.save(join(results_dir, "full_x.npy"), np.asarray(full_x_history))
    np.save(join(results_dir, "full_x_controller.npy"), np.asarray(ea.full_x))
    np.save(join(results_dir, "full_f.npy"), np.asarray(ea.full_f))
    shutil.copy2(join(world.temp_dir.name, "Robot.xml"), join(results_dir, "Robot.xml"))

    metadata = {
        "mode": "Archivebody4SO2control",
        "archive_body_path": os.path.abspath(archive_body_path),
        "archive_fitness_path": os.path.abspath(archive_fitness_path),
        "archive_size": int(np.asarray(archive_fitness).shape[0]),
        "archive_best_idx": int(best_idx),
        "archive_best_fitness": float(best_archive_fitness),
        "fixed_body_raw": fixed_body_raw.tolist(),
        "fixed_body_lengths_m": ((fixed_body_raw + 1.0) / 4.0 + 0.1).tolist(),
        "controller": "SO2Controller",
        "controller_params": int(world.n_controller_params),
        "body_params": int(world.n_body_params),
        "full_genotype_params": int(world.n_full_params),
        "optimized_params": int(world.n_params),
        "so2_controller_param_scale": float(world.controller_param_scale),
        "num_generations": int(num_generations),
        "population_size": int(population_size),
        "n_parents": int(n_parents),
        "n_repeats": int(n_repeats),
        "n_steps": int(n_steps),
        "random_seed": int(random_seed),
        "best_fitness": np.asarray(ea.f_best_so_far, dtype=float).tolist(),
        "best_scalar": float(np.asarray(ea.f_best_so_far, dtype=float).sum()),
    }
    with open(join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Archivebody4SO2control Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Archive body      : {os.path.abspath(archive_body_path)}\n")
        f.write(f"Archive fitness   : {os.path.abspath(archive_fitness_path)}\n")
        f.write(f"Archive best idx  : {best_idx}\n")
        f.write(f"Archive best fit  : {best_archive_fitness:.6f}\n")
        f.write(f"SO2 scale         : {world.controller_param_scale}\n")
        f.write(f"Optimized params  : {world.n_params}\n")
        f.write(f"Saved full params : {world.n_full_params}\n\n")
        labels = ["flat", "ice", "hill"]
        for label, val in zip(labels, np.asarray(ea.f_best_so_far, dtype=float)):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'sum':<6}: {float(np.asarray(ea.f_best_so_far).sum()):10.2f}\n")
    print(f"\nArchivebody4SO2control summary saved to: {score_path}")

    try:
        plot_fitness(ea.full_f, results_dir, OBJECTIVE_LABELS)
        plot_pareto_fronts_3d(
            ea.full_f,
            results_dir,
            OBJECTIVE_LABELS,
            num_generations=len(ea.full_f),
            population_size=population_size,
        )
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MICRO-515 final-project training modes."
    )
    parser.add_argument(
        "--mode",
        choices=["full_so2", "MLPbody4SO2control", "Archivebody4SO2control"],
        default="full_so2",
        help="Training mode. full_so2 preserves the existing body+SO2 co-evolution.",
    )
    parser.add_argument("--num-generations", type=int, default=70)
    parser.add_argument("--population-size", type=int, default=250)
    parser.add_argument("--n-parents", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--mutation-prob", type=float, default=0.3)
    parser.add_argument("--crossover-prob", type=float, default=0.9)
    parser.add_argument("--ckpt-interval", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--mlp-body-checkpoint",
        default=join(ROOT_DIR, "results", "AntHill-v0", "single"),
        help="Challenge3 MLP checkpoint directory or x_best.npy for MLPbody4SO2control.",
    )
    parser.add_argument("--mlp-body-hidden-size", type=int, default=16)
    parser.add_argument(
        "--archive-body",
        default=ARCHIVE_BODY_PATH,
        help="Archive body .npy path for Archivebody4SO2control.",
    )
    parser.add_argument(
        "--archive-fitness",
        default=ARCHIVE_FITNESS_PATH,
        help="Archive fitness .npy path for Archivebody4SO2control.",
    )
    parser.add_argument(
        "--so2-controller-param-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to SO2 genes during fixed-body SO2 training. "
            "Use 1.0 for final_project_test.py raw compatibility; use 0.1 to "
            "match the original FinalWorld training scale."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_parents = args.n_parents if args.n_parents is not None else args.population_size

    common = dict(
        num_generations=args.num_generations,
        population_size=args.population_size,
        n_parents=n_parents,
        n_repeats=args.n_repeats,
        n_steps=args.n_steps,
        mutation_prob=args.mutation_prob,
        crossover_prob=args.crossover_prob,
        ckpt_interval=args.ckpt_interval,
        random_seed=args.random_seed,
    )

    if args.mode == "MLPbody4SO2control":
        run_mlp_body4so2control(
            **common,
            results_dir=args.results_dir
            or join(ROOT_DIR, "results", "MLPbody4SO2control"),
            mlp_body_checkpoint=args.mlp_body_checkpoint,
            mlp_body_hidden_size=args.mlp_body_hidden_size,
            so2_controller_param_scale=args.so2_controller_param_scale,
        )
    elif args.mode == "Archivebody4SO2control":
        run_archive_body4so2control(
            **common,
            results_dir=args.results_dir
            or join(ROOT_DIR, "results", "Archivebody4SO2control"),
            archive_body_path=args.archive_body,
            archive_fitness_path=args.archive_fitness,
            so2_controller_param_scale=args.so2_controller_param_scale,
        )
    else:
        run_multi_task_evolution(
            **common,
            results_dir=args.results_dir
            or join(ROOT_DIR, "results", "final_project_so2_climb"),
            resume=False,
        )


if __name__ == "__main__":
    main()
