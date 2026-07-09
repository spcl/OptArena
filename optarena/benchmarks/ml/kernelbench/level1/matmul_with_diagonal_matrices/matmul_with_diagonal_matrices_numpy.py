import numpy as np


def init():
    pass

def forward(A, B):
    return (np.expand_dims(A, axis=1) * B)
