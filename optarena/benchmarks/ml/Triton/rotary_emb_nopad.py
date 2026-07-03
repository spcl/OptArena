import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'rotary_embedding': 'unsupported Triton wrapper pattern'}

def rotary_embedding(q, k, cos, sin, k_cache, block_tables, kv_lengths):
    raise NotImplementedError('unsupported Triton wrapper pattern')
