import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def chunk_global_reversed_cumsum_scalar(s, dtype):
    return np.flip(np.cumsum(np.flip(s, axis=-1), axis=-1), axis=-1)
