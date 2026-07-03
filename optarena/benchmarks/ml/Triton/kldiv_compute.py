import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def kldivergence(x, y):
    target = np.maximum(y, 1e-12)
    return target * (np.log(target) - x)
