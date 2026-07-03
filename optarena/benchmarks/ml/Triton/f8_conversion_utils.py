import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'f8_to_f16': 'unsupported Triton wrapper pattern', 'f16_to_f8': 'unsupported Triton wrapper pattern'}

def f8_to_f16(x, dtypes):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def f16_to_f8(x, dtypes):
    raise NotImplementedError('unsupported Triton wrapper pattern')
