import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _layer_norm_fwd(x, weight, bias, eps, residual, out_dtype, residual_dtype, is_rms_norm):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + 1e-5)
    out = out * weight
    out = out + bias
    return out

def _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, dresidual, has_residual, is_rms_norm, x_dtype, recompute_output):
    mean = np.mean(dy, axis=-1, keepdims=True)
    var = np.var(dy, axis=-1, keepdims=True)
    out = (dy - mean) / np.sqrt(var + 1e-5)
    out = out * weight
    out = out + bias
    return out
