import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def launch_mul_kernel(src, BLOCK_SIZE):
    return src * BLOCK_SIZE
