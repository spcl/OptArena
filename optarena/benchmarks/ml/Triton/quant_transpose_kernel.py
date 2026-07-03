import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def quantize_global_transpose(input):
    return np.transpose(input)
