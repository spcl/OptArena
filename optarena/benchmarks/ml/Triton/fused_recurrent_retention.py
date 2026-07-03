import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'fused_recurrent_retention': 'unsupported Triton wrapper pattern'}

def fused_recurrent_retention(q, k, v, initial_state, output_final_state):
    raise NotImplementedError('unsupported Triton wrapper pattern')
