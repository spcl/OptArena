import numpy as np


def init():
    pass

def forward(x):
    return np.clip(x, (-1.0), 1.0)
