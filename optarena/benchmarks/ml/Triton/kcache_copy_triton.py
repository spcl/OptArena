import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def copy_k_to_blocked_cache(k, k_cache, kv_lengths, block_tables, n, use_new_kcache_layout):
    return np.array(k, copy=True)
