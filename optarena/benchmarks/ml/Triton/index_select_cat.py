import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'index_select_cat_fwd': 'unsupported Triton wrapper pattern'}

def index_select_cat_fwd(output, source, index):
    raise NotImplementedError('unsupported Triton wrapper pattern')
