import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'flash_attn_triton': 'unsupported Triton wrapper pattern'}

def flash_attn_triton(q, k, v, causal, sm_scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')
