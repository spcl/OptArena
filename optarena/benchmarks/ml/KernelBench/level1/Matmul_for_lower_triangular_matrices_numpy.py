import numpy as np


def init():
    pass

def forward(A, B):
    return np.tril(np.matmul(A, B))
