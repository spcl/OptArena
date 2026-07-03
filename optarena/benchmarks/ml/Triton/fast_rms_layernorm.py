import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def fast_rms_layernorm(layernorm, X, gemma):
    mean = np.mean(layernorm, axis=-1, keepdims=True)
    var = np.var(layernorm, axis=-1, keepdims=True)
    out = (layernorm - mean) / np.sqrt(var + 1e-5)
    return out
