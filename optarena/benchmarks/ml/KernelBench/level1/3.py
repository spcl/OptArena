import numpy as np

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.matmul(A, B)

