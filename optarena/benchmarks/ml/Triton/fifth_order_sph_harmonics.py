import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'calculate_lastdim_num_blocks': 'unsupported Triton wrapper pattern'}

def calculate_lastdim_num_blocks(input_tensor, block_size):
    raise NotImplementedError('unsupported Triton wrapper pattern')
