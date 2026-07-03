import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_configs_io_bound': 'wrapper has no numpy-callable input arguments'}

def init_to_zero(name):
    return np.maximum(name, 0)

def get_configs_io_bound():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def linear_layer(x, weight, bias, activation, act_inputs):
    return np.maximum(x, 0)
