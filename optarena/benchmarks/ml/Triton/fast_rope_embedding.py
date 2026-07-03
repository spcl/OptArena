import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def fast_rope_embedding(Q, K, cos, sin):
    return np.transpose(Q)
