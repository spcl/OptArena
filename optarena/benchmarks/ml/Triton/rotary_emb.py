import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'rotary_emb_fwd': 'unsupported Triton wrapper pattern'}

def rotary_emb_fwd(q, k, cos, sin, partial_rotary_factor):
    raise NotImplementedError('unsupported Triton wrapper pattern')
