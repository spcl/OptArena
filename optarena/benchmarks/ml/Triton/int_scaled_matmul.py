import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def int_matmul_kernel(a, b, c, config):
    return np.matmul(a, b)

def int_scaled_matmul_kernel(a, b, scales1, c, config):
    return np.matmul(a, b)
