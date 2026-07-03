import numpy as np

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

class Model:
    def __init__(self):
        pass

    def forward(self, x):
        norm = np.linalg.norm(x, axis=None, keepdims=False)
        return (x / norm)

