import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_autotune_config': 'wrapper has no numpy-callable input arguments'}

def get_autotune_config():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def matmul(a, b):
    return np.matmul(a, b)
