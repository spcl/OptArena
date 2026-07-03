import numpy as np

batch_size = 4096
dim = 393216

class Model:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        return np.where((x) > 0, (x), (self.alpha) * (np.exp(x) - 1.0))

