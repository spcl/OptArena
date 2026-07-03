import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'fused_recurrent_hgrn': 'unsupported Triton wrapper pattern'}

def fused_recurrent_hgrn(x, g, initial_state, output_final_state):
    raise NotImplementedError('unsupported Triton wrapper pattern')
