import numpy as np

batch_size = 32768
dim = 65535

class Model:
    def __init__(self):
        pass

    def forward(self, x):
        return (x / np.linalg.norm(x, axis=1, keepdims=True))

