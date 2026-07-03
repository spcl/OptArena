import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'triton_f4_to_bf16': 'unsupported Triton wrapper pattern'}

def triton_f4_to_bf16(x):
    raise NotImplementedError('unsupported Triton wrapper pattern')
