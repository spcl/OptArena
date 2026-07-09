import numpy as np


def init():
    pass

def forward(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
