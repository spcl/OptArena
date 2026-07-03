import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'quantize_global': 'unsupported Triton wrapper pattern'}

def quantize_global(x):
    raise NotImplementedError('unsupported Triton wrapper pattern')
