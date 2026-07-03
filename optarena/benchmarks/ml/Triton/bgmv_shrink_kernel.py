import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'_bgmv_shrink': 'unsupported Triton wrapper pattern'}

def _bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling):
    raise NotImplementedError('unsupported Triton wrapper pattern')
