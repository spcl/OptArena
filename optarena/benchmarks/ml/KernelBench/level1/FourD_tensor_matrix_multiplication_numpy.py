import numpy as np


def init():
    pass

def forward(A, B):
    return np.einsum('bijl,lk->bijk', A, B)
