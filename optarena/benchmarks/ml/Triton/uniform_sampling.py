import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'philox_cuda_seed_offset': 'unsupported Triton wrapper pattern', 'volume': 'unsupported Triton wrapper pattern', 'uniform_': 'unsupported Triton wrapper pattern'}

def philox_cuda_seed_offset(increment, device):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def volume(shape):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def uniform_(from_, to):
    raise NotImplementedError('unsupported Triton wrapper pattern')
