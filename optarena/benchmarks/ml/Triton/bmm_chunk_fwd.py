import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def _bmm_chunk_fwd(a, b, chunk_size, seq_idx, causal, output_dtype):
    return np.matmul(a, b)
