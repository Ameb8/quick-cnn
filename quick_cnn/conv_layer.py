import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size, stride, padding):
        # Creator defines convolution dimensions
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        # Randomly initializes weights and biases of filters
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
        self.filter_biases = np.random.normal(0, 0.1, num_filters)
        self.last_input = None

    def get_output_dim(self, input_size):
        return (input_size - self.filter_size + 2 * self.padding) // self.stride

    def get_output_size(self, input):
        """
        :param input: data being processed
        :return: height x width of output feature map
        """
        input_h, input_w = input.shape
        output_h = self.get_output_dim(input_h)
        output_w = self.get_output_dim(input_w)
        return output_h, output_w

    def forward_pass(self, input_data):
        """
        Conducts forward pass of convolution layer
        :param pixel_data: ndarray containing pixel data from image
        :return: feature map output as ndarray
        """
        # Determine size of feature map and initialize
        input_h, input_w = input_data.shape
        output_h, output_w = self.get_output_size(input_data)
        feature_map = np.empty((self.num_filters, output_h, output_w), dtype=float)

        # Pad the input image if padding is specified
        if self.padding > 0:
            input_data = np.pad(input_data, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)

        self.last_input = input_data.copy()

        for f in range(self.num_filters): # iterate through each filter
            filter = self.filters[f]
            bias = self.filter_biases[f]

            for i in range(output_h):
                for j in range(output_w):
                    # Calculate top left corner of image section
                    top_left_x = i * self.stride
                    top_left_y = j * self.stride

                    # Extract the patch from the input data
                    patch = input_data[top_left_x:top_left_x + self.filter_size, top_left_y:top_left_y + self.filter_size]

                    # Calculate feature map value for image patch
                    feature_map[f, i, j] = np.sum(patch * filter) + bias

        # Remove padding from input data
        if self.padding > 0:
            input_data = input_data[self.padding:self.padding + input_h, self.padding:self.padding + input_w]

        # Apply ReLU
        feature_map = np.maximum(0, feature_map)

        return feature_map

    def backward_pass(self, d_out, learning_rate):
        """
        Performs backwards pass through convolutional layer
        :param d_out: gradient of loss of output feature map (shape: num_filters x output_h x output_w)
        :param learning_rate: learning rate for updating weights
        :return: gradient of loss w.r.t. input to this layer
        """
        input_data = self.last_input
        d_input = np.zeros_like(input_data)
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.filter_biases)

        _, output_h, output_w = d_out.shape

        for f in range(self.num_filters):
            for i in range(output_h):
                for j in range(output_w):
                    # Top-left corner of patch
                    top_left_x = i * self.stride
                    top_left_y = j * self.stride

                    # Get the input patch
                    patch = input_data[top_left_x:top_left_x + self.filter_size,
                            top_left_y:top_left_y + self.filter_size]

                    # Update gradients
                    d_filters[f] += patch * d_out[f, i, j]
                    d_biases[f] += d_out[f, i, j]
                    d_input[top_left_x:top_left_x + self.filter_size, top_left_y:top_left_y + self.filter_size] += \
                    self.filters[f] * d_out[f, i, j]

        # Remove padding from d_input if applied in forward
        if self.padding > 0:
            d_input = d_input[self.padding:-self.padding, self.padding:-self.padding]

        # Update parameters
        self.filters -= learning_rate * d_filters
        self.filter_biases -= learning_rate * d_biases

        return d_input


def test_conv_layer():
    np.random.seed(42)  # for reproducibility

    # Create a dummy 8x8 grayscale "image"
    input_data = np.random.randn(8, 8)

    # Create a ConvLayer with 3 filters, 3x3 kernel size, stride 1, and padding 1
    conv = ConvLayer(num_filters=3, filter_size=3, stride=1, padding=1)

    # Forward pass
    output = conv.forward_pass(input_data)
    print("Forward pass output shape:", output.shape)
    print("Forward pass output:", output)

    # Create a dummy gradient output (same shape as output)
    d_out = np.random.randn(*output.shape)

    # Backward pass
    d_input = conv.backward_pass(d_out, learning_rate=0.01)
    print("Backward pass output shape:", d_input.shape)
    print("Backward pass output:", d_input)

# Run the test
test_conv_layer()
