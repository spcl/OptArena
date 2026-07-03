import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'can_use_int32_index': 'unsupported Triton wrapper pattern'}

def can_use_int32_index(tensor):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def argmax(inp, dim, keepdim):
    values = np.max(inp, axis=dim, keepdims=keepdim) if 'keepdim' in globals() else np.max(inp, axis=dim)
    indices = np.argmax(inp, axis=dim)
    return values, indices
