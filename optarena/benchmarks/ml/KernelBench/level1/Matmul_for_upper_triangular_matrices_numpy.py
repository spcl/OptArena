import numpy as np


def init():
    pass

def forward(A, B):
    return np.triu(np.matmul(A, B))
