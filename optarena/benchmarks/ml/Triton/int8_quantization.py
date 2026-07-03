import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'per_block_int8': 'unsupported Triton wrapper pattern'}

def per_block_int8(q, k, BLKQ, BLKK):
    raise NotImplementedError('unsupported Triton wrapper pattern')
