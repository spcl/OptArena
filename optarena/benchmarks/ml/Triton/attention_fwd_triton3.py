import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'_forward': 'unsupported Triton wrapper pattern'}

def _forward(q, k, v, sm_scale, o, m, l, end, sliding_window, init, complement_sliding_window):
    raise NotImplementedError('unsupported Triton wrapper pattern')
