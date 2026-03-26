import numpy as np


class OscillatoryController:
    """Simple oscillatory controller using sine waves for each actuator."""

    def __init__(
        self, input_size: int = 0, output_size: int = None, hidden_size: int = 0
    ):
        assert output_size is not None, (
            "output_size must be specified for OscillatoryController"
        )

        self.output_size = output_size
        self.time_step = 0.0
        self.n_params = self.get_num_params()

        # Parameters: [amplitudes, frequencies, phases] for each actuator
        self.amplitudes = np.random.uniform(0.1, 1.0, self.output_size)
        self.frequencies = np.random.uniform(0.5, 2.0, self.output_size)
        self.phases = np.random.uniform(0, 2 * np.pi, self.output_size)

    def get_action(self, state):
        """Generate oscillatory actions based on time.

        Args:
            state: Observation from environment. Can be shape (obs_size,) or (n_envs, obs_size)
                  for vectorized environments. Not used by this controller but needed for API.

        Returns:
            actions: Shape matches input - (action_size,) or (n_envs, action_size)
        """
        # Determine if we're dealing with batched observations
        is_batched = len(state.shape) > 1 if state is not None else False
        batch_size = state.shape[0] if is_batched else 1

        # Simple sine wave oscillations
        actions = self.amplitudes * np.sin(
            2 * np.pi * self.frequencies * self.time_step + self.phases
        )
        self.time_step += 0.01  # Increment time

        # Clip to valid action range
        actions = np.clip(actions, -1.0, 1.0)

        # Handle batched observations (for vectorized environments)
        if is_batched:
            # Return same action for all environments in the batch
            actions = np.tile(actions, (batch_size, 1))

        return actions

    def set_weights(self, weights):
        """Set controller parameters from flat array."""
        # Weights = [amplitudes, frequencies, phases]
        self.amplitudes = weights[: self.output_size]
        self.frequencies = 5 * weights[self.output_size : 2 * self.output_size]
        self.phases = np.pi * weights[2 * self.output_size : 3 * self.output_size]
        self.time_step = 0  # Reset time

    def geno2pheno(self, genotype):
        """Alias for set_weights."""
        self.set_weights(genotype)
        self.reset_controller()

    def get_num_params(self):
        """Return total number of parameters."""
        return 3 * self.output_size

    def reset_controller(self):
        self.time_step = 0.0