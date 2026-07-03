import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'token_att_fwd2': 'unsupported Triton wrapper pattern'}

def token_att_fwd2(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen):
    raise NotImplementedError('unsupported Triton wrapper pattern')
