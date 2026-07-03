import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def rmsnorm_wrapper(x, rms_weights, eps):
    out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    out = out * rms_weights
    return out
