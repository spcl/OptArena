import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'update_fn': 'unsupported Triton wrapper pattern'}

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    raise NotImplementedError('unsupported Triton wrapper pattern')
