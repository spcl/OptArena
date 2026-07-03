import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def triton_mul2(x, BLOCK_SIZE):
    return x * BLOCK_SIZE

def triton_mul2_inplace(x, BLOCK_SIZE):
    return x * BLOCK_SIZE
