import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def fused_native_layer_norm(primals_1, primals_2, primals_3):
    mean = np.mean(primals_1, axis=-1, keepdims=True)
    var = np.var(primals_1, axis=-1, keepdims=True)
    out = (primals_1 - mean) / np.sqrt(var + 1e-5)
    return out
