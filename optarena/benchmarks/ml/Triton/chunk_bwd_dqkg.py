import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'chunk_bwd_dqkg_fn': 'unsupported Triton wrapper pattern'}

def chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')
