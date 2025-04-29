import numpy as np

class SoftmaxLossLayer:
    def __init__(self):
        self.probs = None     # softmax probabilities

    def forward_pass(self, input_logits, labels=None):
        """
        input_logits: raw scores from last dense layer (shape: num_classes)
        labels: optional; if given, computes loss
        """
        # Compute probabilities
        shifted_logits = input_logits - np.max(input_logits)  # for numerical stability
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores)

        # Calculate loss if labels passed
        if labels is not None:
            self.labels = labels
            loss = -np.sum(labels * np.log(self.probs + 1e-9))  # add epsilon for stability
            return loss

        return self.probs

    def forward_pass(self, input_logits):
        """
        input_logits: raw scores from last dense layer (shape: num_classes)
        """
        shifted_logits = input_logits - np.max(input_logits)  # for numerical stability
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores)

        return self.probs

    def backward_pass(self, labels):
        """
        Gradient of cross-entropy loss w.r.t input logits, assuming softmax applied
        labels: true labels of the data
        """
        return self.probs - labels

    def compute_loss(self, labels, learn_rate):
        """
        Function to compute cross-entropy loss.
        labels: true labels of the data
        """
        loss = -np.sum(labels * np.log(self.probs + 1e-9))  # add epsilon for stability
        return loss



