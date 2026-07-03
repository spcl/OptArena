import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'rbe_triton_wrapper': 'unsupported Triton wrapper pattern'}

def rbe_triton_wrapper(x, pos):
    raise NotImplementedError('unsupported Triton wrapper pattern')
