import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'chunk_fwd_intra_gated_gk_fn': 'unsupported Triton wrapper pattern', 'chunk_fwd_o_gated_gk_fn': 'unsupported Triton wrapper pattern'}

def chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, BT, scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')
