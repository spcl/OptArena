import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'embedding': 'unsupported Triton wrapper pattern'}

def embedding(input_ids, weight, vob_start_id, vob_end_id, out):
    raise NotImplementedError('unsupported Triton wrapper pattern')
