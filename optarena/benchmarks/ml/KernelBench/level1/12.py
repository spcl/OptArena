import numpy as np

M = 4096
N = 4096

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return (np.expand_dims(A, axis=1) * B)

