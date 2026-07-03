import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'contiguous': 'unsupported Triton wrapper pattern'}

def contiguous(fn):
    raise NotImplementedError('unsupported Triton wrapper pattern')
