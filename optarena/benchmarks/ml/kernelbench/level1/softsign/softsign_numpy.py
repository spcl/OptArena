import numpy as np


def init():
    pass

def forward(x):
    return (x / (1 + np.abs(x)))
