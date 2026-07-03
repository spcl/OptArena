import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'cross_entropy_bwd': 'cross entropy backward requires source-specific gradient translation'}

def cross_entropy_fwd(logits, labels, smoothing, logit_scale, lse_square_scale, ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    labels = np.asarray(labels, dtype=np.int64)
    labels = np.reshape(labels, log_probs.shape[:-1])
    return -np.take_along_axis(log_probs, np.expand_dims(labels, axis=-1), axis=-1).squeeze(axis=-1)

def cross_entropy_bwd(dloss, logits, lse, labels, smoothing, logit_scale, lse_square_scale, ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING):
    raise NotImplementedError('cross entropy backward requires source-specific gradient translation')
