import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'context_attention_fwd_ppl_int8kv': 'unsupported Triton wrapper pattern'}

def context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len):
    raise NotImplementedError('unsupported Triton wrapper pattern')
