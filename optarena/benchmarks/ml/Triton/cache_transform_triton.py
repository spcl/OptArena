import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'get_xine_cache': 'unsupported Triton wrapper pattern'}

def get_xine_cache(lengths, cos_cache, sin_cache, is_prompts):
    raise NotImplementedError('unsupported Triton wrapper pattern')
