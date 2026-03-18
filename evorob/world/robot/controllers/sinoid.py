import numpy as np

from evorob.world.robot.controllers.base import Controller

class OscillatoryController(Controller):
    """Simple oscillatory controller using sine waves for each actuator.

    This controller generates periodic motion patterns without using observations.
    Each joint oscillates with its own amplitude, frequency, and phase.
    """

    def __init__(
        self, input_size: int = 0, output_size: int = None, hidden_size: int = 0
    ):
        """Initialize the oscillatory controller.

        Args:
            output_size: Number of actuators to control
            input_size: Not used, kept for API compatibility
        """
        assert output_size is not None, (
            "output_size must be specified for OscillatoryController"
        )
        self.output_size = output_size
        self.time_step = 0.0
        self.dt = 0.05
        # Speed bias: push actions closer to full range without hard clipping.
        self.drive_gain = 1.6
        # Deterministic phase template breaks symmetry at CMA-ES mean x0=0.
        self.phase_template = np.linspace(-np.pi, np.pi, self.output_size, endpoint=False)
        self.n_params = self.get_num_params()

        # TODO: Initialize parameters for oscillatory control
        # You need 3 parameters per actuator: amplitude, frequency, phase
        # - self.amplitudes: uniform random in [0.1, 1.0] (shape: output_size)
        # - self.frequencies: uniform random in [0.5, 2.0] (shape: output_size)
        # - self.phases: uniform random in [0, 2*pi] (shape: output_size)
        # Hint: Use np.random.uniform(low, high, size)
        self.amplitudes = np.random.uniform(0.1, 1.0, self.output_size)
        self.frequencies = np.random.uniform(0.5, 2.0, self.output_size)
        self.phases = np.random.uniform(-np.pi, np.pi, self.output_size)

    def get_action(self, state):
        """Generate oscillatory actions based on time.

        Args:
            state: Observation (not used by this controller)

        Returns:
            actions: Array of actuator commands, shape (output_size,) or (batch_size, output_size)
        """
        # TODO: Compute oscillatory actions using sine waves
        # Compute action with parameterized sine wave.
        # Then increment self.time_step by e.g. 0.01
        # Clip actions to [-1.0, 1.0]
        #
        # For vectorized environments (batch of observations):
        # Check if state is 2D, if so replicate actions for each environment
        # Hint: Use np.tile(actions, (batch_size, 1))
        state = np.asarray(state)
        theta = 2 * np.pi * self.frequencies * self.time_step + self.phases
        # Add a small second harmonic to enrich gaits without adding parameters.
        oscillation = (np.sin(theta) + 0.25 * np.sin(2.0 * theta)) / 1.25
        raw_actions = self.drive_gain * self.amplitudes * oscillation
        self.time_step += self.dt
        actions = np.tanh(raw_actions)
        if state.ndim == 2:
            actions = np.tile(actions, (state.shape[0], 1))
        return actions

    def set_weights(self, weights):
        """Set controller parameters from flat array.

        Args:
            weights: Flat array of size (3 * output_size,)
                    [amplitudes, frequencies, phases]
        """
        # TODO: Extract parameters from weights array
        # weights structure: [amp1, amp2, ..., freq1, freq2, ..., phase1, phase2, ...]
        # Update self.amplitudes, self.frequencies, self.phases accordingly
        # Reset time to 0
        weights = np.asarray(weights)
        if weights.size != 3 * self.output_size:
            raise ValueError(f"Expected weights size {3 * self.output_size}, got {weights.size}")
        raw_amp = weights[: self.output_size]
        raw_freq = weights[self.output_size : 2 * self.output_size]
        raw_phase = weights[2 * self.output_size :]

        # Speed-oriented mapping:
        # - amplitude in [0.15, 1.00] to avoid very weak gaits
        # - frequency in [0.8, 5.5] Hz with center around 2.2 Hz
        # - phase as template + bounded delta (keeps coordinated legs)
        self.amplitudes = 0.15 + 0.85 * (0.5 * (np.tanh(raw_amp) + 1.0))
        self.frequencies = np.clip(2.2 * np.exp(0.30 * raw_freq), 0.8, 5.5)
        phase_delta = 0.75 * np.pi * np.tanh(raw_phase)
        self.phases = np.angle(np.exp(1j * (self.phase_template + phase_delta)))
        self.reset_controller()

    def geno2pheno(self, genotype):
        """Alias for set_weights."""
        self.set_weights(genotype)
        self.reset_controller()

    def get_num_params(self):
        """Return total number of parameters.

        Returns:
            int: 3 * output_size (amplitude, frequency, phase for each actuator)
        """
        # TODO: Return the total number of parameters
        return 3 * self.output_size

    def reset_controller(self):
        """Reset the controller state (time)."""
        self.time_step = 0.0
