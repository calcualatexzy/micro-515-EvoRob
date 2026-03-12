import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        """Initialize a simple feedforward neural network.

        Network structure: input -> hidden -> output
        Activation: tanh on both layers

        Args:
            input_size: Dimension of input (observation size)
            output_size: Dimension of output (action size)
            hidden_size: Number of hidden neurons
        """
        # Here we randomly initialize our neural network layers,
        # as well as our input and output size.
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        # TODO: Initialize weight matrices with uniform random values in [-1, 1]
        # - self.input_to_hidden: shape (hidden_size, input_size)
        # - self.hidden_to_output: shape (output_size, hidden_size)
        # Hint: Use np.random.uniform(-1, 1, (rows, cols))
        self.input_to_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.hidden_to_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.hidden_bias = np.zeros(hidden_size)
        self.output_bias = np.zeros(output_size)

        # TODO: Compute number of parameters in each layer
        self.n_params_i2h = hidden_size * input_size
        self.n_params_h2o = output_size * hidden_size
        self.n_params_b1 = hidden_size
        self.n_params_b2 = output_size

        self.n_params = self.get_num_params()


    def get_action(self, state):
        """Forward pass through the network.

        Args:
            state: Observation array, shape (input_size,) or (batch_size, input_size)

        Returns:
            action: Output array, shape (output_size,) or (batch_size, output_size)
        """
        # TODO: Perform forward pass computation
        # 1. Hidden layer: hidden = ...
        # 2. Output layer: output = ...
        # 3. Clip output to [-1, 1] using np.clip()
        #
        # Hint: Use @ operator or np.matmul for matrix multiplication
        # Hint: .T transposes a matrix
        # Hint: np.tanh() applies tanh element-wise
        state = np.asarray(state)
        # hidden = np.tanh(state @ self.input_to_hidden.T)
        hidden = np.tanh(state @ self.input_to_hidden.T + self.hidden_bias)
        # output = np.tanh(hidden @ self.hidden_to_output.T)
        output = np.tanh(hidden @ self.hidden_to_output.T + self.output_bias)
        return np.clip(output, -1.0, 1.0)

    def set_weights(self, encoding):
        """Set network weights from a flat parameter vector.

        Args:
            encoding: Flat array of size (n_params,) containing all weights
        """
        # TODO: Map the flat encoding to weight matrices
        # 1. Split encoding into two parts:
        #    - First n_params_i2h values for input_to_hidden
        #    - Remaining n_params_h2o values for hidden_to_output
        # 2. Reshape each part to match the weight matrix shapes
        #
        # Hint: Use array slicing: encoding[:n] and encoding[n:]
        # Hint: Use np.reshape(array, (rows, cols)) or array.reshape((rows, cols))
        encoding = np.asarray(encoding)
        if encoding.size != self.n_params:
            raise ValueError(f"Expected encoding size {self.n_params}, got {encoding.size}")

        idx = 0
        i2h_end = idx + self.n_params_i2h
        self.input_to_hidden = encoding[idx:i2h_end].reshape((self.n_hidden, self.n_input))
        idx = i2h_end

        h2o_end = idx + self.n_params_h2o
        self.hidden_to_output = encoding[idx:h2o_end].reshape((self.n_output, self.n_hidden))
        idx = h2o_end

        b1_end = idx + self.n_params_b1
        self.hidden_bias = encoding[idx:b1_end]
        idx = b1_end

        b2_end = idx + self.n_params_b2
        self.output_bias = encoding[idx:b2_end]

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        # To provide a genetic encoding for our neural network controller,
        # we compute and store the number of parameters in our NN class.
        # TODO: Return the total number of parameters in both layers!
        return self.n_params_i2h + self.n_params_h2o + self.n_params_b1 + self.n_params_b2

    def reset_controller(self, batch_size=1) -> None:
        pass
