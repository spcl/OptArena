import numpy as np


def init(dim):
    pass

def forward(x, dim):
    return np.sum(x, axis=dim, keepdims=True)
