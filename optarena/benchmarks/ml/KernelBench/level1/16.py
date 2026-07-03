import numpy as np

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.matmul(A.T, B)

