import numpy as np

M = 16384 * 2
N = 16 * 2

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.matmul(A, B)

