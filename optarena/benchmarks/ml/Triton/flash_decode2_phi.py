import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'flash_decode_stage2': 'unsupported Triton wrapper pattern'}

def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, Out, block_seq):
    raise NotImplementedError('unsupported Triton wrapper pattern')
