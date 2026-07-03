import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def mv(inp, vec):
    return np.matmul(inp, vec)
