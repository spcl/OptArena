import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _swiglu_fwd(xy, out):
    return (xy / (1.0 + np.exp(-xy))) * out
