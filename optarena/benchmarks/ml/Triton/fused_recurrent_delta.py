import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'fused_recurrent_delta_rule': 'unsupported Triton wrapper pattern'}

def fused_recurrent_delta_rule(q, k, v, beta, scale, initial_state, output_final_state):
    raise NotImplementedError('unsupported Triton wrapper pattern')
