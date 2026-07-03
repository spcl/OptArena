import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'chunk_fwd_h_fn': 'unsupported Triton wrapper pattern', 'chunk_fwd_o_fn': 'unsupported Triton wrapper pattern', 'chunk_bwd_dh_fn': 'unsupported Triton wrapper pattern', 'chunk_bwd_dqkv_fn': 'unsupported Triton wrapper pattern', 'chunk_retention': 'unsupported Triton wrapper pattern'}

def chunk_fwd_h_fn(k, v, BT, initial_state, output_final_state):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def chunk_fwd_o_fn(h, q, k, v, BT, scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def chunk_bwd_dh_fn(do, q, k, v, BT, scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def chunk_bwd_dqkv_fn(do, q, k, v, h, dh, scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def chunk_retention(q, k, v, initial_state, output_final_state, scale, checkpoint_level):
    raise NotImplementedError('unsupported Triton wrapper pattern')
