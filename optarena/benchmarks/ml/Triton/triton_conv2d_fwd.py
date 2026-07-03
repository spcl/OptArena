import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'conv2d_forward': 'unsupported Triton wrapper pattern'}

def conv2d_forward(input_tensor, weight_tensor, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, groups, fp16, tf32):
    raise NotImplementedError('unsupported Triton wrapper pattern')
