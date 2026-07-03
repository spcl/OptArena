import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'_sgmv_expand_slice': 'unsupported Triton wrapper pattern'}

def _sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor, lora_indices_tensor, batches, max_seq_length, token_nums, slice_offset, slice_size, add_inputs):
    raise NotImplementedError('unsupported Triton wrapper pattern')
