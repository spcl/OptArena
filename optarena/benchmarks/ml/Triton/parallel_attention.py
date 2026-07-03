import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'parallel_rebased': 'unsupported Triton wrapper pattern'}

def parallel_rebased(q, k, v, eps, use_scale, use_normalize, return_both):
    raise NotImplementedError('unsupported Triton wrapper pattern')
