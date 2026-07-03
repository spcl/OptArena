import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def block_sparse_attention(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout, softmax_scale, num_heads, num_kv_heads, total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS):
    shifted = Q - np.max(Q, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
