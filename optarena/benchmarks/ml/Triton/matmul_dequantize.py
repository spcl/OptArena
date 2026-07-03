import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'dequantize_int4': 'unsupported Triton wrapper pattern'}

def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, group_size, output):
    return np.matmul(x, qweight)

def matmul_dequantize_int4_s2(x, qweight, scales, qzeros, group_size, output):
    return np.matmul(x, qweight)

def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def matmul_dequantize_int4_s1(a, b, b_scale, b_zero_point, group_size, out):
    return np.matmul(a, b)
