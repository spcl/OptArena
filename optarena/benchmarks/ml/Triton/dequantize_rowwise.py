import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'dequantize_rowwise': 'unsupported Triton wrapper pattern'}

def dequantize_rowwise(x, state_x):
    raise NotImplementedError('unsupported Triton wrapper pattern')
