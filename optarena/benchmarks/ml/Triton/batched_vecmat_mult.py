import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'batched_vecmat': 'unsupported Triton wrapper pattern'}

def batched_vecmat(M, N, K, block_m, block_n, block_k, num_warps, num_stages):
    raise NotImplementedError('unsupported Triton wrapper pattern')
