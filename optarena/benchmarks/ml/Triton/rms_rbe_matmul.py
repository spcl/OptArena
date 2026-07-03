import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def rms_matmul_rbe_wrapper(x, weight, rms_w, use_rbe, start_pos, n_heads, head_dim):
    out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    out = out * weight
    return out
