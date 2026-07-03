import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'contiguous': 'unsupported Triton wrapper pattern', 'chunk_retention': 'unsupported Triton wrapper pattern'}

def contiguous(fn):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def chunk_retention(q, k, v, initial_state, output_final_state):
    raise NotImplementedError('unsupported Triton wrapper pattern')
