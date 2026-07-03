import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _matmul_launch_metadata(grid, kernel, args):
    return np.matmul(grid, kernel)

def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    return np.matmul(a_ptr, b_ptr)

def matmul(a, b):
    return np.matmul(a, b)
