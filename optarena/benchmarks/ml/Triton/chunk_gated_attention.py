import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'fwd_pre': 'unsupported Triton wrapper pattern', 'fwd_inner': 'unsupported Triton wrapper pattern'}

def fwd_pre(g, B, H, T, S, BT):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def fwd_inner(q, k, v, g, B, H, T, K, V, BT, BK, BV, gatek, h0, ht):
    raise NotImplementedError('unsupported Triton wrapper pattern')
