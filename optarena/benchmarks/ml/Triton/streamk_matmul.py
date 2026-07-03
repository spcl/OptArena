import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'swizzle_tile': 'unsupported Triton wrapper pattern', 'linear_tile': 'unsupported Triton wrapper pattern', 'mac_loop': 'unsupported Triton wrapper pattern', 'first_wave': 'unsupported Triton wrapper pattern', 'full_tiles': 'unsupported Triton wrapper pattern'}

def swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_iter, end_iter, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def first_wave(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def full_tiles(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_tiles_streamk, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    raise NotImplementedError('unsupported Triton wrapper pattern')
