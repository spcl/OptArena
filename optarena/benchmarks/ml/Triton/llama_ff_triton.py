import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'kernel_ff': 'unsupported Triton wrapper pattern'}

def kernel_ff(x, w1, w3, rms_w):
    raise NotImplementedError('unsupported Triton wrapper pattern')
