import numpy as np

batch_size = 4096
dim = 393216

class Model:
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def forward(self, x):
        return np.where((x) > 0, (x), (self.negative_slope) * (x))

