import numpy as np

N = 16
M = 1024
K = 2048
L = 768

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.matmul(A, B)

