import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'decoding_fused_rotary_embedding': 'unsupported Triton wrapper pattern'}

def decoding_fused_rotary_embedding(q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths, use_new_kcache_layout):
    raise NotImplementedError('unsupported Triton wrapper pattern')
