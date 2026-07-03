import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    return np.transpose(b)

def matmul_dequantize_int4_s1(a, b, b_scale, b_zero_point, group_size, out):
    return np.matmul(a, b)

def quantize_int4(weight, group_size, tp_rank):
    return np.transpose(weight)
