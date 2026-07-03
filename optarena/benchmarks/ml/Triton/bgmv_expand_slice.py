import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'_bgmv_expand_slice': 'unsupported Triton wrapper pattern'}

def _bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size, add_inputs):
    raise NotImplementedError('unsupported Triton wrapper pattern')
