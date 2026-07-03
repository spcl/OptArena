import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def rms_norm_kernel(Y, X, W, y_stride_r, y_stride_c, x_stride_r, x_stride_c, N, eps, BLOCK_SIZE):
    out = Y / np.sqrt(np.mean(Y * Y, axis=-1, keepdims=True) + 1e-6)
    return out

def rms_norm(x, normalized_shape, weight, eps):
    out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    out = out * weight
    return out
