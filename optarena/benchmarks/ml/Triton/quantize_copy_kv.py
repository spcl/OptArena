import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    return np.array(K, copy=True)
