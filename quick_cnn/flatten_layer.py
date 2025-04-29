import numpy as np

class FlattenLayer:
    def __init__(self):
        self.input_shape = None;

    def forward_pass(self, input_data):
        self.input_shape = input_data.shape
        return input_data.reshape(-1)

    def backward_pass(self, d_out, learn_rate):
        return d_out.reshape(self.input_shape)
