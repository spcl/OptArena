import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_configs_io_bound': 'wrapper has no numpy-callable input arguments'}

def get_configs_io_bound():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias):
    return np.matmul(a, b)
