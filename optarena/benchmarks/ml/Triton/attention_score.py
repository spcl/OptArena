import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_score': 'unsupported Triton wrapper pattern'}

def get_score(q, k, m, sliding_window, complement_sliding_window):
    raise NotImplementedError('unsupported Triton wrapper pattern')
