import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'dim_compress': 'unsupported Triton wrapper pattern'}

def dim_compress(inp, dims):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def mean_dim(x, dim, keepdim):
    return np.mean(x)
