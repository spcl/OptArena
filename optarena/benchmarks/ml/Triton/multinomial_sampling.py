import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_kernel_meta': 'unsupported Triton wrapper pattern'}

def get_kernel_meta(tensor):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def multinomial_sampling(scores, seeds, offsets, indices):
    return scores * seeds
