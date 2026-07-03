import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'triton_f4_to_scaled_bf16': 'unsupported Triton wrapper pattern'}

def triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size):
    raise NotImplementedError('unsupported Triton wrapper pattern')
