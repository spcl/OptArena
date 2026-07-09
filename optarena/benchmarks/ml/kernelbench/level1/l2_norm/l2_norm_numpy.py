import numpy as np


def init():
    pass

def forward(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True))
