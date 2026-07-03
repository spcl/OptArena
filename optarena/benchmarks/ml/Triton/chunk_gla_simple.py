import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'chunk_fwd_o_fn': 'unsupported Triton wrapper pattern'}

def chunk_fwd_o_fn(h, q, k, v, g, BT, scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')
