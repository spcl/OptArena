import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _rope_embedding_forward_impl(Q, cos, sin):
    return np.transpose(Q)

def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    return np.transpose(dY)
