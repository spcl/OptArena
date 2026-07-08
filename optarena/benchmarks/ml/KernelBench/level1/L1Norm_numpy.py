import numpy as np


def init():
    pass

def forward(x):
    return (x / np.mean(np.abs(x), axis=1, keepdims=True))
