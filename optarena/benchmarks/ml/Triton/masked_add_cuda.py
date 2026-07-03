import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def masked_add(grad, p_data, p_mask, alpha):
    return grad + p_data
