import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    target = np.maximum(y_true, 1e-12)
    return target * (np.log(target) - y_pred)

def kldiv_backward_triton(target, grad_output, new_grads, log_target):
    target = np.maximum(grad_output, 1e-12)
    return target * (np.log(target) - target)
