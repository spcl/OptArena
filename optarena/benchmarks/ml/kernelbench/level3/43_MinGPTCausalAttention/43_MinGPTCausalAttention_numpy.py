import numpy as np

def _softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _split_size(x, split_size, axis=0):
    indices = list(range(split_size, x.shape[axis], split_size))
    return tuple(np.split(x, indices, axis=axis))

def init(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
    global c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias, attn_dropout, resid_dropout, bias
    c_attn_weight = np.zeros((3 * n_embd, n_embd), dtype=np.float32)
    c_attn_bias = np.zeros((3 * n_embd,), dtype=np.float32) if True else np.zeros((3 * n_embd,), dtype=np.float32)
    c_proj_weight = np.zeros((n_embd, n_embd), dtype=np.float32)
    c_proj_bias = np.zeros((n_embd,), dtype=np.float32) if True else np.zeros((n_embd,), dtype=np.float32)
    attn_dropout = None
    resid_dropout = None
    bias = np.reshape(np.tril(np.ones((max_seqlen, max_seqlen), dtype=np.float32)), (1, 1, max_seqlen, max_seqlen))

def forward(x, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
    (B, T, C) = x.shape
    (q, k, v) = _split_size(((x) @ c_attn_weight.T + c_attn_bias), n_embd, axis=2)
    k = np.swapaxes(np.reshape(k, (B, T, n_head, (C // n_head))), 1, 2)
    q = np.swapaxes(np.reshape(q, (B, T, n_head, (C // n_head))), 1, 2)
    v = np.swapaxes(np.reshape(v, (B, T, n_head, (C // n_head))), 1, 2)
    att = ((q @ np.swapaxes(k, (-2), (-1))) * (1.0 / np.sqrt(k.shape[(-1)])))
    att = np.where((bias[:, :, :T, :T] == 0), float('-inf'), att)
    att = _softmax(att, axis=(-1))
    att = att
    y = (att @ v)
    y = np.reshape(np.swapaxes(y, 1, 2), (B, T, C))
    y = ((y) @ c_proj_weight.T + c_proj_bias)
    return y

