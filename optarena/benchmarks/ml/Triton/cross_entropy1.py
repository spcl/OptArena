import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def cross_entropy_loss(logits, labels, label_smoothing, lse_square_scale, ignored_index, inplace_backward, process_group):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    labels = np.asarray(labels, dtype=np.int64)
    labels = np.reshape(labels, log_probs.shape[:-1])
    return -np.take_along_axis(log_probs, np.expand_dims(labels, axis=-1), axis=-1).squeeze(axis=-1)
