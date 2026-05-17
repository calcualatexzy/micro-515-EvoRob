import numpy as np

from evorob.world.robot.controllers.base import Controller


def RK45(state, A, dt):
    """Runge-Kutta 4th/5th order integration."""
    A1 = np.matmul(A, state)
    A2 = np.matmul(A, (state + dt / 2 * A1))
    A3 = np.matmul(A, (state + dt / 2 * A2))
    A4 = np.matmul(A, (state + dt * A3))
    return state + dt / 6 * (A1 + 2 * (A2 + A3) + A4)


class SO2Controller(Controller):

    def __init__(self, input_size: int,  output_size: int, hidden_size: int):
        """
        SO2 oscillator Controller. [https://www.nature.com/articles/s41467-024-50131-4]
        - Uses internal oscillators coupled via a weight matrix.
        - Uses a small evolved linear sensory-feedback layer to modulate the
          oscillator action output while preserving rhythmic CPG dynamics.
        """
        num_dofs = output_size
        dt = 0.05
        inter_con_density = 0.5
        feedback_scale = 0.1
        self.controller_type = "SO2"
        self.dt = dt
        self.num_dofs = num_dofs
        self.feedback_scale = feedback_scale

        # Initialize network structure (consistent randomness)
        weight_matrix, weight_map, weights = self.initalise_network(num_dofs, inter_con_density)

        self.A = weight_matrix
        self.weight_map = weight_map
        self.weights = weights
        self.oscillator_weight_mask = (
            (self.weight_map[:, 0] % 2 == 0)
            & (self.weight_map[:, 1] == self.weight_map[:, 0] + 1)
        )

        # Dimensions
        self.n_weights = len(self.weights)
        self.n_initial_state_params = num_dofs * 2
        self.n_output_gain_params = num_dofs
        self.n_output_bias_params = num_dofs
        self.n_feedback_inputs = 5
        self.n_feedback_params = self.n_feedback_inputs * num_dofs
        self.n_params = (
            self.n_weights
            + self.n_initial_state_params
            + self.n_output_gain_params
            + self.n_output_bias_params
            + self.n_feedback_params
        )

        # For compatibility with Controller inspection
        self.n_input = input_size
        self.n_output = num_dofs  # Actions
        if self.n_input < 18:
            raise ValueError(
                "Closed-loop SO2 expects the default ant observation with at "
                f"least 18 values, got input size {self.n_input}."
            )

        # Internal state shape: (2*N, 1) or (2*N, Batch)
        self.state_shape = (num_dofs * 2, 1)

        # Default initial state
        self.template_initial_state = np.ones(self.state_shape) * np.sqrt(2) / 2
        self.output_gain = np.ones(num_dofs)
        self.output_bias = np.zeros(num_dofs)
        self.feedback_weights = np.zeros((self.n_feedback_inputs, self.n_output))

        # Current running state (will be set in reset_controller)
        self.y = None

    def initalise_network(self, num_dofs, inter_con_density: float = 0.5):
        """
        Initializes the network topology.
        Fixed seed ensures structure is random but consistent across runs.
        """
        rng = np.random.default_rng(42)

        # 1. Intrinsic Oscillator Weights (2*pi frequency)
        oscillator_weights = np.ones(num_dofs) * 2 * np.pi

        n_states = num_dofs * 2
        weight_matrix = np.zeros((n_states, n_states))

        # Place intrinsic weights on super-diagonal
        rows_osc = np.arange(0, n_states, 2)
        cols_osc = np.arange(1, n_states, 2)
        weight_matrix[rows_osc, cols_osc] = oscillator_weights

        # 2. Random Inter-connections
        n_possible = (num_dofs * (num_dofs - 1)) // 2
        n_active = int(np.round(n_possible * inter_con_density))

        if n_active > 0:
            triu_rows, triu_cols = np.triu_indices(num_dofs, k=1)
            perm_indices = rng.permutation(len(triu_rows))[:n_active]

            selected_rows = triu_rows[perm_indices]
            selected_cols = triu_cols[perm_indices]

            inter_weights = rng.random(n_active)

            # Map DOF indices to State indices (Position-to-Position coupling)
            weight_matrix[selected_rows * 2, selected_cols * 2] = inter_weights

        # 3. Extract Genotype & Apply Anti-Symmetry
        weight_map = np.argwhere(weight_matrix > 0)
        genotype = weight_matrix[weight_map[:, 0], weight_map[:, 1]]
        weight_matrix -= weight_matrix.T

        return weight_matrix, weight_map, genotype

    def geno2pheno(self, genotype: np.ndarray) -> None:
        """
        Map the flat genotype vector to the weight matrix A and initial states.
        """
        genotype = np.asarray(genotype, dtype=float)
        if genotype.size != self.n_params:
            raise ValueError(
                f"Expected SO2 genotype size {self.n_params}, got {genotype.size}. "
                "Retrain or update the checkpoint to match the current closed-loop "
                "SO2 controller."
            )

        raw_weights = np.asarray(genotype[:self.n_weights], dtype=float)

        # The EA uses the same [-1, 1] bounds for controller and body genes.
        # Map raw oscillator-frequency genes to a useful positive range around
        # the default 2*pi frequency, while leaving inter-oscillator couplings
        # in the raw signed range.
        self.genotype = raw_weights.copy()
        self.genotype[self.oscillator_weight_mask] = (
            2 * np.pi * (1.0 + raw_weights[self.oscillator_weight_mask])
        )

        idx = self.n_weights

        # Update initial oscillator state template.
        state_end = idx + self.n_initial_state_params
        self.template_initial_state = genotype[idx:state_end].reshape(self.state_shape)
        idx = state_end

        # Per-joint output gain and bias make the open-loop oscillator much more
        # useful for hill climbing: the EA can evolve stronger knees/hips and a
        # slight forward-leaning action offset without changing the oscillator
        # topology.  Raw genes stay in [-1, 1].
        gain_end = idx + self.n_output_gain_params
        raw_gain = np.asarray(genotype[idx:gain_end], dtype=float)
        self.output_gain = 0.25 + 1.75 * (raw_gain + 1.0) / 2.0  # [0.25, 2.0]
        idx = gain_end

        bias_end = idx + self.n_output_bias_params
        raw_bias = np.asarray(genotype[idx:bias_end], dtype=float)
        self.output_bias = 0.5 * raw_bias  # [-0.5, 0.5]
        idx = bias_end

        # Linear sensory-feedback term.  Observations are tanh-compressed in
        # get_action() to keep large velocities from dominating the oscillator.
        feedback_end = idx + self.n_feedback_params
        raw_feedback = np.asarray(genotype[idx:feedback_end], dtype=float)
        self.feedback_weights = raw_feedback.reshape(
            (self.n_feedback_inputs, self.n_output)
        )

        # Update weight matrix (Upper Triangle)
        self.A[self.weight_map[:, 0], self.weight_map[:, 1]] = self.genotype
        # Update weight matrix (Lower Triangle) - Enforce Anti-Symmetry
        self.A[self.weight_map[:, 1], self.weight_map[:, 0]] = -self.genotype

    def reset_controller(self, batch_size=1) -> None:
        """
        Resets internal states for a batch of environments.
        """
        # Tile the initial state to match the batch size: (2*N, Batch)
        self.y = np.tile(self.template_initial_state, (1, batch_size))

    def _feedback_features(self, state: np.ndarray) -> np.ndarray:
        """Extract 5 compact posture/progress features from the ant observation.

        Observations are qpos[2:] followed by qvel:
          [z, qw, qx, qy, qz, joint_pos..., x_vel, y_vel, z_vel, roll_rate,
           pitch_rate, yaw_rate, joint_vel...]

        The feedback features are:
          torso height, torso roll, torso pitch, forward velocity, pitch rate.
        """
        z = state[:, 0]
        qw = state[:, 1]
        qx = state[:, 2]
        qy = state[:, 3]
        qz = state[:, 4]
        x_velocity = state[:, 13]
        pitch_rate = state[:, 17]

        roll = np.arctan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        pitch_arg = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(pitch_arg)

        return np.column_stack([
            z - 0.5,
            roll,
            pitch,
            x_velocity,
            pitch_rate,
        ])

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the next action based on internal oscillator state.

        :param state: Observations (Batch, Input_Dim). Tanh-compressed and fed
                      through an evolved linear layer to close the SO2 loop.
        :return: Actions (Batch, Output_Dim)
        """
        state = np.asarray(state, dtype=float)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if state.shape[1] != self.n_input:
            raise ValueError(
                f"Expected observation size {self.n_input}, got {state.shape[1]}"
            )

        next_y = RK45(self.y, self.A, self.dt)
        self.y = next_y
        oscillator_actions = (
            self.output_gain[:, None] * next_y[1::2, :] + self.output_bias[:, None]
        )
        feedback_state = np.tanh(self._feedback_features(state))
        feedback_actions = feedback_state @ self.feedback_weights * self.feedback_scale
        return np.tanh(oscillator_actions.T + feedback_actions)
