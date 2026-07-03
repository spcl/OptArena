import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def destindex_copy_kv(K, DestLoc, Out):
    return np.array(K, copy=True)
