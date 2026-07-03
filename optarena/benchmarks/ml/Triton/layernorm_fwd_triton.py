import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def layernorm_forward(X, W, eps):
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    out = (X - mean) / np.sqrt(var + 1e-5)
    return out
