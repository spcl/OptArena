import numpy as np

def _split_size(x, split_size, axis=0):
    indices = list(range(split_size, x.shape[axis], split_size))
    return tuple(np.split(x, indices, axis=axis))

def init(n_embd, n_head, max_seqlen):
    global c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias, bias
    c_attn_weight = np.zeros((3 * n_embd, n_embd), dtype=np.float32)
    c_attn_bias = np.zeros((3 * n_embd,), dtype=np.float32) if True else np.zeros((3 * n_embd,), dtype=np.float32)
    c_proj_weight = np.zeros((n_embd, n_embd), dtype=np.float32)
    c_proj_bias = np.zeros((n_embd,), dtype=np.float32) if True else np.zeros((n_embd,), dtype=np.float32)
    bias = np.reshape(np.tril(np.ones((max_seqlen, max_seqlen), dtype=np.float32)), (1, 1, max_seqlen, max_seqlen))

def forward(x, n_embd, n_head, max_seqlen):
    (B, T, C) = x.shape
    (q, k, v) = _split_size(((x) @ c_attn_weight.T + c_attn_bias), n_embd, axis=2)
    k = np.swapaxes(np.reshape(k, (B, T, n_head, (C // n_head))), 1, 2)
    q = np.swapaxes(np.reshape(q, (B, T, n_head, (C // n_head))), 1, 2)
    v = np.swapaxes(np.reshape(v, (B, T, n_head, (C // n_head))), 1, 2)
    att = ((q @ np.swapaxes(k, (-2), (-1))) * (1.0 / np.sqrt(k.shape[(-1)])))
    att = np.where((bias[:, :, :T, :T] == 0), float('-inf'), att)
    att = np.maximum(att, 0)
    y = (att @ v)
    y = np.reshape(np.swapaxes(y, 1, 2), (B, T, C))
    return y

