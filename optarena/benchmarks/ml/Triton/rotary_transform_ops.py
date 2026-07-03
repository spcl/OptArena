import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'apply_rotary': 'unsupported Triton wrapper pattern'}

def apply_rotary(x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, interleaved, inplace, conjugate):
    raise NotImplementedError('unsupported Triton wrapper pattern')
