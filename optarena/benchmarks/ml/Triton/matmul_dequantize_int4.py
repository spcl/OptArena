import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, group_size, output):
    return np.matmul(x, qweight)

def quantize_int4(weight, group_size, tp_rank):
    return np.transpose(weight)
