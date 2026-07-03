import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def chunk_global_cumsum_vector(s, dtype):
    return np.cumsum(s, axis=-1)
