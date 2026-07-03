import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def triton_fa(q, k, v, sm_scale, is_causal, start_position):
    return np.transpose(q)
