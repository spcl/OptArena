import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    return np.array(KV_nope, copy=True)
