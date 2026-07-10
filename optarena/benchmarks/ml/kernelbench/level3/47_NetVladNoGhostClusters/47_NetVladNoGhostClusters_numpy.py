import numpy as np

def _batch_norm(x, weight, bias, running_mean, running_var, eps):
    shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    return (x - running_mean.reshape(shape)) / np.sqrt(running_var.reshape(shape) + eps) * weight.reshape(shape) + bias.reshape(shape)


def _softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def init(cluster_size, feature_size, ghost_clusters):
    global clusters, batch_norm_weight, batch_norm_bias, batch_norm_running_mean, batch_norm_running_var, batch_norm_eps, clusters2, out_dim
    clusters = np.array(((1 / np.sqrt(feature_size)) * np.zeros((feature_size, (cluster_size + ghost_clusters)), dtype=np.float32)), dtype=np.float32)
    batch_norm_weight = np.ones((cluster_size + ghost_clusters,), dtype=np.float32)
    batch_norm_bias = np.zeros((cluster_size + ghost_clusters,), dtype=np.float32)
    batch_norm_running_mean = np.zeros((cluster_size + ghost_clusters,), dtype=np.float32)
    batch_norm_running_var = np.ones((cluster_size + ghost_clusters,), dtype=np.float32)
    batch_norm_eps = 1e-5
    clusters2 = np.array(((1 / np.sqrt(feature_size)) * np.zeros((1, feature_size, cluster_size), dtype=np.float32)), dtype=np.float32)
    out_dim = (cluster_size * feature_size)

def forward(x, mask, cluster_size, feature_size, ghost_clusters):
    max_sample = x.shape[1]
    x = np.reshape(x, ((-1), feature_size))
    assignment = np.matmul(x, clusters)
    assignment = _batch_norm(assignment, batch_norm_weight, batch_norm_bias, batch_norm_running_mean, batch_norm_running_var, batch_norm_eps)
    assignment = _softmax(assignment, axis=1)
    assignment = assignment[:, :cluster_size]
    assignment = np.reshape(assignment, ((-1), max_sample, cluster_size))
    a_sum = np.sum(th, axis=1, keepdims=True)
    a = (a_sum * clusters2)
    assignment = np.swapaxes(assignment, 1, 2)
    x = np.reshape(x, ((-1), max_sample, feature_size))
    vlad = np.matmul(assignment, x)
    vlad = np.swapaxes(vlad, 1, 2)
    vlad = (vlad - a)
    vlad = ((vlad) / np.maximum(np.linalg.norm(vlad, axis=1, keepdims=True), 1e-12))
    vlad = np.reshape(vlad, ((-1), (cluster_size * feature_size)))
    vlad = ((vlad) / np.maximum(np.linalg.norm(vlad, axis=1, keepdims=True), 1e-12))
    return vlad

