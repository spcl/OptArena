import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'is_cuda': 'wrapper has no numpy-callable input arguments', 'get_cuda_autotune_config': 'wrapper has no numpy-callable input arguments', 'get_hip_autotune_config': 'wrapper has no numpy-callable input arguments', 'get_autotune_config': 'wrapper has no numpy-callable input arguments'}

def is_cuda():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def get_cuda_autotune_config():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def get_hip_autotune_config():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def get_autotune_config():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def matmul(a, b, activation):
    return np.maximum(a, 0)
