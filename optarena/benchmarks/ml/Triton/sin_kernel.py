import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'call_kernel': 'unsupported Triton wrapper pattern'}

def call_kernel(x):
    raise NotImplementedError('unsupported Triton wrapper pattern')
