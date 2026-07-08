import numpy as np


def init(dim):
    pass

def forward(x, dim):
    return np.cumsum(x, axis=dim)
