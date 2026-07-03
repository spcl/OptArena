import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def fused_add_mul_activation_torch(in_out_tensor, bias, in_tensor):
    return np.maximum(in_out_tensor, 0)
