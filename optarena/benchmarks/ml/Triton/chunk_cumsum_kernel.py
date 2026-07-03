import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def chunk_global_cumsum_scalar(s, dtype):
    return np.cumsum(s, axis=-1)
