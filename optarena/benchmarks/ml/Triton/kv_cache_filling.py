import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_kernel_meta': 'unsupported Triton wrapper pattern', 'fill_kv_cache': 'unsupported Triton wrapper pattern'}

def get_kernel_meta(tensor):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def fill_kv_cache(k_states, v_states, k_caches, v_caches, q_start_loc, q_seq_length, kv_seq_length, max_q_seq_length, block_offsets, k_scales_zeros, v_scales_zeros, quant_policy):
    raise NotImplementedError('unsupported Triton wrapper pattern')
