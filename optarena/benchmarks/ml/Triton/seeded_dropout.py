import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def seeded_dropout(x, p, seed):
    return np.array(x, copy=True)
