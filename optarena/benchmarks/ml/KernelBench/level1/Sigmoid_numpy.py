import numpy as np


def init():
    pass

def forward(x):
    return (1.0 / (1.0 + np.exp(-(x))))
