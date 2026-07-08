import numpy as np


def init(alpha=1.0):
    pass

def forward(x, alpha):
    return np.where((x) > 0, (x), (alpha) * (np.exp(x) - 1.0))
