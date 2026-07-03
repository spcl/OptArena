import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'index_select_cat_bwd': 'unsupported Triton wrapper pattern'}

def index_select_cat_bwd(grad_source, index, grad_output):
    raise NotImplementedError('unsupported Triton wrapper pattern')
