import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'context_attention_fwd': 'unsupported Triton wrapper pattern'}

def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    raise NotImplementedError('unsupported Triton wrapper pattern')
