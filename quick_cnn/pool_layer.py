import numpy as np

class PoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None
        self.max_indices = None

    def get_output_size(self, input_size):
        return (input_size - self.pool_size) // self.stride + 1

    def forward_pass(self, input_data):
        input_h, input_w = input_data.shape
        output_h = self.get_output_size(input_h)
        output_w = self.get_output_size(input_w)

        # Create an empty output feature map
        pooled_output = np.empty((output_h, output_w), dtype=float)

        # Store the indices of the max values for backpropagation
        self.max_indices = np.zeros((output_h, output_w, 2), dtype=int)

        for i in range(output_h):
            for j in range(output_w):
                # Calculate the top-left corner of the pooling window
                top_left_x = i * self.stride
                top_left_y = j * self.stride

                # Get the pooling window (submatrix)
                patch = input_data[top_left_x:top_left_x + self.pool_size, top_left_y:top_left_y + self.pool_size]

                # Find the max value and its position in the window
                max_val = np.max(patch)
                max_pos = np.unravel_index(np.argmax(patch), patch.shape)

                # Save the max position for backward pass
                self.max_indices[i, j] = (top_left_x + max_pos[0], top_left_y + max_pos[1])

                # Set the pooled output
                pooled_output[i, j] = max_val

        # Save the input for backpropagation
        self.last_input = input_data

        return pooled_output

    def backward_pass(self, d_out, learn_rate):
        input_data = self.last_input
        d_input = np.zeros_like(input_data, dtype=float)

        # Backpropagate the gradients using the max positions recorded
        output_h, output_w = d_out.shape

        for i in range(output_h):
            for j in range(output_w):
                # Get the max position for this output element
                max_pos_x, max_pos_y = self.max_indices[i, j]

                print(f"Max position for output element ({i}, {j}): {max_pos_x}, {max_pos_y}")  # Debug print
                print(f"Gradient value at output[{i}, {j}]: {d_out[i, j]}")  # Debug print

                # Only propagate the gradient to the max position
                d_input[max_pos_x, max_pos_y] += d_out[i, j]

        print(f"Gradient w.r.t Input (Backward Pass): \n{d_input}")  # Debug print
        return d_input
