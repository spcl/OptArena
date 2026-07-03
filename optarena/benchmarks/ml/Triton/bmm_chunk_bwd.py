import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _bmm_chunk_bwd(a, dout, residual, out):
    return np.matmul(a, dout)
