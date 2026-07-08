import numpy as np


def init():
    pass

def forward(x):
    return np.clip(((x) + 3.0) / 6.0, 0.0, 1.0)
