import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'chunk_fwd_h_fn': 'unsupported Triton wrapper pattern'}

def chunk_fwd_h_fn(k, w, u, BT, initial_state, final_state):
    raise NotImplementedError('unsupported Triton wrapper pattern')
