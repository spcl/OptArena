import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'forward': 'unsupported Triton wrapper pattern'}

def forward(q, k, v, q_scale, k_scale):
    raise NotImplementedError('unsupported Triton wrapper pattern')
