import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'context_attention_fwd': 'unsupported Triton wrapper pattern'}

def context_attention_fwd(q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs):
    raise NotImplementedError('unsupported Triton wrapper pattern')
