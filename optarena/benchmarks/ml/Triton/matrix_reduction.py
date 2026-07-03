import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def load_reduce(BLOCK_M, BLOCK_N, dtype_str):
    return np.matmul(BLOCK_M, BLOCK_N)
