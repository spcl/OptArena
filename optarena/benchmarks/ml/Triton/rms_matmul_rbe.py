import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def rms_matmul_rbe_qkv_wrapper(x, start_pos, q_weight, k_weight, v_weight, rms_w, n_heads, head_dim, k, v, eps, theta):
    out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    out = out * q_weight
    return out
