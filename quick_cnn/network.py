from quick_cnn.conv_layer import ConvLayer
from quick_cnn.linear_layer import LinearLayer
from quick_cnn.pool_layer import PoolLayer


class ImageClassifier:
    def __init__(self, input_h, input_w, classes):
        self.input_h = input_h
        self.input_w = input_w
        self.classes = classes
        self.layers = []

    def add_conv_layer(self, num_filters, filter_size, stride, padding, max_pool=True):
        self.layers.append(ConvLayer(num_filters, filter_size, stride, padding))

        if max_pool:
            self.layers.append(PoolLayer(2, 2))

    def add_linear_layer(self, input_size, output_size):
        self.layers.append(LinearLayer(input_size, output_size))

    def make_inference(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_pass(input_data)

        return input_data

    def train(self, train_data, train_label, epochs, learn_rate, verbose=True):
        loss_per_epoch = []

        for epoch in range(epochs):
            total_loss = 0

            for data, label in zip(train_data, train_label):
                # Calculate probabilities and loss
                probs = self.make_inference(data)
                total_loss += self.layers[-1].compute_loss(label)

                # Backward pass
                d_out = self.layers[-1].backward_pass(label)
                for layer in reversed(self.layers[:-1]):
                    d_out = layer.backward_pass(d_out, learn_rate)

            loss_per_epoch.append(total_loss)
            print(f'Epoch: {epoch+1}/{epochs}, Loss: {total_loss}\n')

        return loss_per_epoch
