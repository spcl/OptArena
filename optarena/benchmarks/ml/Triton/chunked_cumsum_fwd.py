import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias, dt_softplus, dt_limit):
    return np.cumsum(dt, axis=-1)
