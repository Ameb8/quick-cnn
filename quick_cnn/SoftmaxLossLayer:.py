import numpy as np

class SoftmaxLossLayer:
    def __init__(self):
        self.probs = None     # softmax probabilities
        self.labels = None    # ground-truth labels

    def forward_pass(self, input_logits, labels):
        """
        input_logits: raw scores from last dense layer (shape: num_classes)
        labels: one-hot encoded true labels, same shape
        """
        self.labels = labels
        shifted_logits = input_logits - np.max(input_logits)  # for numerical stability
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores)

        # Cross-entropy loss
        loss = -np.sum(labels * np.log(self.probs + 1e-9))  # add epsilon for stability
        return loss

    def backward_pass(self):
        """
        Gradient of cross-entropy loss w.r.t input logits, assuming softmax applied
        """
        return self.probs - self.labels

