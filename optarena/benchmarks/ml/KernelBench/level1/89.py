import numpy as np

batch_size = 32768
input_shape = (32768,)
dim = 1

class Model:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        return np.cumsum(x, axis=self.dim)

