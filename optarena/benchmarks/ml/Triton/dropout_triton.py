import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def dropout(x, x_keep, p):
    return np.array(x, copy=True)
