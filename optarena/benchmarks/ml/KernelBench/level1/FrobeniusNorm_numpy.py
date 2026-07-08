import numpy as np


def init():
    pass

def forward(x):
    norm = np.linalg.norm(x, axis=None, keepdims=False)
    return (x / norm)
