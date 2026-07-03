import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def geglu_forward(a, b):
    gate = 0.5 * a * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * (a ** 3))))
    return gate * b

def geglu_backward(a, b, dc):
    gate = 0.5 * a * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * (a ** 3))))
    return gate * b
