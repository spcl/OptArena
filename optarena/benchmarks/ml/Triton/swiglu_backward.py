import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _swiglu_bwd(xy, dout, dxy, recompute_output, out):
    return (xy / (1.0 + np.exp(-xy))) * dout
