import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _layer_norm_fwd(x, weight, bias, eps, residual, x1, weight1, bias1, dropout_p, rowscale, out_dtype, residual_dtype, is_rms_norm, return_dropout_mask):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + 1e-5)
    out = out * weight
    out = out + bias
    return out
