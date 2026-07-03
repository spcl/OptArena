import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def max(inp):
    return np.max(inp)

def max_dim(inp, dim, keepdim):
    values = np.max(inp, axis=dim, keepdims=keepdim) if 'keepdim' in globals() else np.max(inp, axis=dim)
    indices = np.argmax(inp, axis=dim)
    return values, indices
