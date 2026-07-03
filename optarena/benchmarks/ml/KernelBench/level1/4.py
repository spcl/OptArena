import numpy as np

M = 256 * 8
K = 131072 * 8

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.matmul(A, B)

