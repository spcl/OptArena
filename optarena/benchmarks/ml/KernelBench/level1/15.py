import numpy as np

M = 4096

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.tril(np.matmul(A, B))

