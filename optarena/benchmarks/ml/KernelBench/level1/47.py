import numpy as np

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

class Model:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        return np.sum(x, axis=self.dim, keepdims=True)

