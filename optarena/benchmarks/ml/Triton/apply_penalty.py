import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'apply_penalty': 'unsupported Triton wrapper pattern'}

def apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
    raise NotImplementedError('unsupported Triton wrapper pattern')
