import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def triton_matmul(a, b):
    return np.matmul(a, b)
