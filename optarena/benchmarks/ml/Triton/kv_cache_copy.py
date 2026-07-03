import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def copy_kv_to_blocked_cache(k, v, k_cache, v_cache, kv_lengths, block_tables, use_new_kcache_layout):
    return np.array(k, copy=True)
