import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def matmul_dequantize_int8(a, b, b_scale, out):
    return np.matmul(a, b)
