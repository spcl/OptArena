import numpy as np

M = 8205
K = 2949
N = 5921

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.matmul(A, B)

