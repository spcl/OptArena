import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'heuristics_for_tile_size': 'unsupported Triton wrapper pattern', 'heuristics_for_num_warps': 'unsupported Triton wrapper pattern', 'pow_func_scalar_tensor_wrapper_rank_1': 'wrapper has no numpy-callable input arguments'}

def heuristics_for_tile_size(max_tile_size):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def heuristics_for_num_warps(tile_size):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def pow_func_scalar_tensor_wrapper_rank_1():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def pow_func_scalar_tensor_kernel_rank_1(val0, in0_ptr, out0_ptr, in0_stride0, in0_stride_order0, out0_stride0, out0_stride_order0, s0, num_tasks, tiles_per_cta, tile_size0, one_tile_per_cta):
    return np.power(val0, in0_ptr)
