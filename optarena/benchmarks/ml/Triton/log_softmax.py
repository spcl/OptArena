import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def log_softmax(x, dim, dtype):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    log_den = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_den
