import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights with small random values and biases with zeros
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)

        # Placeholders for backward pass
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward_pass(self, input_data):
        """
        input_data: ndarray of input data (size: input_size)
        """
        self.input = input_data
        output = np.dot(input_data, self.weights) + self.biases
        return output

    def backward_pass(self, d_out, learn_rate):
        """
        grad_output: gradient flowing from the next layer (shape: output_size)
        returns: gradient w.r.t input to this layer
        """
        # Gradient w.r.t weights and biases
        self.grad_weights = np.outer(self.input, d_out)
        self.grad_biases = d_out

        # Gradient w.r.t input
        grad_input = np.dot(d_out, self.weights.T)
        return grad_input
