import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def rope_forward(q, k, cos, sin):
    return np.transpose(q)
