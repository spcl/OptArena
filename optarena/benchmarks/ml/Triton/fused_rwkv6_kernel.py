import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'fused_recurrent_rwkv6': 'unsupported Triton wrapper pattern'}

def fused_recurrent_rwkv6(r, k, v, w, u, scale, initial_state, output_final_state):
    raise NotImplementedError('unsupported Triton wrapper pattern')
