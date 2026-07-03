import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'spinning_lock': 'unsupported Triton wrapper pattern'}

def spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):
    raise NotImplementedError('unsupported Triton wrapper pattern')
