import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'token_att_fwd': 'unsupported Triton wrapper pattern'}

def token_att_fwd(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    raise NotImplementedError('unsupported Triton wrapper pattern')
