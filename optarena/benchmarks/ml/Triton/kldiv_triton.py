import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_num_warps': 'unsupported Triton wrapper pattern'}

def get_num_warps(BLOCK_SIZE):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def kldiv_forward_triton(y_pred, y_true, log_target, reduction):
    target = np.maximum(y_true, 1e-12)
    return target * (np.log(target) - y_pred)

def kldiv_backward_triton(input, target, grad_output, log_target):
    target = np.maximum(target, 1e-12)
    return target * (np.log(target) - input)
