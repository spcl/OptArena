import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def rmsnorm_triton_wrapper(x, rms_w, eps):
    out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return out
