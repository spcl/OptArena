import numpy as np


def init(dim):
    pass

def forward(x, mask, dim):
    return np.cumsum((x * mask), axis=dim)
