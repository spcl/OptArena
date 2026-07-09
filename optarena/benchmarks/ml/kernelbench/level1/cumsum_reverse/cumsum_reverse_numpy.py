import numpy as np


def init(dim):
    pass

def forward(x, dim):
    return np.flip(np.cumsum(np.flip(x, axis=dim), axis=dim), axis=dim)
