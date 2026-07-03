import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'quantize_int8_perrow': 'unsupported Triton wrapper pattern', 'quantize_int8': 'unsupported Triton wrapper pattern'}

def quantize_int8_perrow(fpa):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def matmul_quantize_int8(fpa, b, b_scale, out):
    return np.matmul(fpa, b)

def matmul_int8(a, a_scale, b, b_scale, out):
    return np.matmul(a, a_scale)

def quantize_int8(weight, axis):
    raise NotImplementedError('unsupported Triton wrapper pattern')
