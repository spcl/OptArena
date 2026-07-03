import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'is_hip': 'wrapper has no numpy-callable input arguments'}

def is_hip():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def swiglu_forward(a, b):
    return (a / (1.0 + np.exp(-a))) * b

def swiglu_backward(a, b, dc):
    return (a / (1.0 + np.exp(-a))) * b
