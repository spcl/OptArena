import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def rope_backward(dq, dk, cos, sin):
    return np.transpose(dq)
