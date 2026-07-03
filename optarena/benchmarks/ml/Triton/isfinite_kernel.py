import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'heuristics_for_tile_size': 'unsupported Triton wrapper pattern', 'heuristics_for_num_warps': 'unsupported Triton wrapper pattern', 'isfinite_func_wrapper_rank_1': 'wrapper has no numpy-callable input arguments'}

def heuristics_for_tile_size(max_tile_size):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def heuristics_for_num_warps(tile_size):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def isfinite_func_wrapper_rank_1():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')
