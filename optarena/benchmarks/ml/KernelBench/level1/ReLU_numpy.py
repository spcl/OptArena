import numpy as np


def init():
    pass

def forward(x):
    return np.maximum(x, 0)
