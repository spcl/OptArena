import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'wrapper_nested3': 'unsupported Triton wrapper pattern'}

def wrapper_nested3(n_rows, n_cols):
    raise NotImplementedError('unsupported Triton wrapper pattern')
