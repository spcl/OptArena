import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'_triton_mixed_sparse_attention': 'unsupported Triton wrapper pattern'}

def _triton_mixed_sparse_attention(q, k, v, seqlens, block_count, block_offset, column_count, column_index, sm_scale, block_size_M, block_size_N):
    raise NotImplementedError('unsupported Triton wrapper pattern')
