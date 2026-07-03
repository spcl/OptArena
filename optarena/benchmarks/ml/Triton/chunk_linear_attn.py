import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'chunk_linear_attn': 'unsupported Triton wrapper pattern'}

def chunk_linear_attn(q, k, v, scale, initial_state, output_final_state, normalize):
    raise NotImplementedError('unsupported Triton wrapper pattern')
