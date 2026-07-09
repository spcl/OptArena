import numpy as np


def init():
    pass

def forward(A, B):
    return np.matmul(A.T, B)
