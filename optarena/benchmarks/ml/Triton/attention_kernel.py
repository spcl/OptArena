import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'_attention_rel_h_rel_w_kernel_aligned_device': 'unsupported Triton wrapper pattern'}

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, o, BLOCK_M, BLOCK_N, num_warps, num_stages):
    raise NotImplementedError('unsupported Triton wrapper pattern')
