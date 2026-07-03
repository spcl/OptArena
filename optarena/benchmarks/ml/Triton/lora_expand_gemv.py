import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'_bgmv_expand': 'unsupported Triton wrapper pattern'}

def _bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor, add_inputs):
    raise NotImplementedError('unsupported Triton wrapper pattern')
