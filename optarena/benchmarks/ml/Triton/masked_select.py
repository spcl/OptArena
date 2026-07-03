import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'broadcastable': 'unsupported Triton wrapper pattern', 'cfggen': 'wrapper has no numpy-callable input arguments', 'masked_select': 'unsupported Triton wrapper pattern'}

def broadcastable(s1, s2):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def cfggen():
    raise NotImplementedError('wrapper has no numpy-callable input arguments')

def masked_select(inp, mask):
    raise NotImplementedError('unsupported Triton wrapper pattern')
