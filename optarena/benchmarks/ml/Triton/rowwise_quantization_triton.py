import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'quantize_rowwise': 'unsupported Triton wrapper pattern'}

def quantize_rowwise(x):
    raise NotImplementedError('unsupported Triton wrapper pattern')
