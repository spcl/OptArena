import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'relu_forward_wrapper_rank_1': 'wrapper has no numpy-callable input arguments'}

def heuristics_for_tile_size(max_tile_size):
    return np.maximum(max_tile_size, 0)

def heuristics_for_num_warps(tile_size):
    return np.maximum(tile_size, 0)

def relu_forward_wrapper_rank_1():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')
