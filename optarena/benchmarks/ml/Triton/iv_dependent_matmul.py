import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def iv_dependent_matmul_wrapper(M, K, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, type, device):
    return np.matmul(type, device)
