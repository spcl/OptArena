import numpy as np


def init(dim):
    pass

def forward(x, dim):
    return np.max(x, axis=dim, keepdims=False)
