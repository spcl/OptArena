import numpy as np

batch_size = 4096
dim = 393216

class Model:
    def __init__(self):
        pass

    def forward(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

