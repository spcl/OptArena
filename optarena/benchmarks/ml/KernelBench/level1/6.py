import numpy as np

M = 256
N = 256
K = 131072 * 4

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.matmul(A, B)

