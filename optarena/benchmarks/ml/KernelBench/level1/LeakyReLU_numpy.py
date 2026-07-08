import numpy as np


def init(negative_slope=0.01):
    pass

def forward(x, negative_slope):
    return np.where((x) > 0, (x), (negative_slope) * (x))
