import numpy as np

def _as_tuple(value, dims):
    if isinstance(value, tuple):
        return value
    return tuple(value for _ in range(dims))


def _batch_norm(x, weight, bias, running_mean, running_var, eps):
    shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    return (x - running_mean.reshape(shape)) / np.sqrt(running_var.reshape(shape) + eps) * weight.reshape(shape) + bias.reshape(shape)


def _conv2d(x, weight, bias, stride, padding, dilation, groups):
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    n, c_in, h, w = x.shape
    c_out, c_per_group, kh, kw = weight.shape
    oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    padded = np.zeros((n, c_in, h + 2 * padding[0], w + 2 * padding[1]), dtype=x.dtype)
    padded[:, :, padding[0]:padding[0] + h, padding[1]:padding[1] + w] = x
    out = np.zeros((n, c_out, oh, ow), dtype=x.dtype)
    out_per_group = c_out // groups
    in_per_group = c_in // groups
    for b in range(n):
        for oc in range(c_out):
            g = oc // out_per_group
            for oy in range(oh):
                for ox in range(ow):
                    total = 0.0
                    for icg in range(c_per_group):
                        ic = g * in_per_group + icg
                        for ky in range(kh):
                            iy = oy * stride[0] + ky * dilation[0]
                            for kx in range(kw):
                                ix = ox * stride[1] + kx * dilation[1]
                                total += padded[b, ic, iy, ix] * weight[oc, icg, ky, kx]
                    out[b, oc, oy, ox] = total + bias[oc]
    return out

def init(in_channels, out_channels, stride=1):
    global conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps, relu, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, bn2_eps, downsample_0_weight, downsample_0_bias, downsample_0_stride, downsample_0_padding, downsample_0_dilation, downsample_0_groups, downsample_1_weight, downsample_1_bias, downsample_1_running_mean, downsample_1_running_var, downsample_1_eps
    conv1_weight = np.zeros((out_channels, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    conv1_bias = np.zeros((out_channels,), dtype=np.float32)
    conv1_stride = stride
    conv1_padding = 1
    conv1_dilation = 1
    conv1_groups = 1
    bn1_weight = np.ones((out_channels,), dtype=np.float32)
    bn1_bias = np.zeros((out_channels,), dtype=np.float32)
    bn1_running_mean = np.zeros((out_channels,), dtype=np.float32)
    bn1_running_var = np.ones((out_channels,), dtype=np.float32)
    bn1_eps = 1e-5
    relu = None
    conv2_weight = np.zeros((out_channels, out_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    conv2_bias = np.zeros((out_channels,), dtype=np.float32)
    conv2_stride = 1
    conv2_padding = 1
    conv2_dilation = 1
    conv2_groups = 1
    bn2_weight = np.ones((out_channels,), dtype=np.float32)
    bn2_bias = np.zeros((out_channels,), dtype=np.float32)
    bn2_running_mean = np.zeros((out_channels,), dtype=np.float32)
    bn2_running_var = np.ones((out_channels,), dtype=np.float32)
    bn2_eps = 1e-5
    downsample_0_weight = np.zeros((out_channels * 1, in_channels // 1) + _as_tuple(1, 2), dtype=np.float32)
    downsample_0_bias = np.zeros((out_channels * 1,), dtype=np.float32)
    downsample_0_stride = stride
    downsample_0_padding = 0
    downsample_0_dilation = 1
    downsample_0_groups = 1
    downsample_1_weight = np.ones((out_channels * 1,), dtype=np.float32)
    downsample_1_bias = np.zeros((out_channels * 1,), dtype=np.float32)
    downsample_1_running_mean = np.zeros((out_channels * 1,), dtype=np.float32)
    downsample_1_running_var = np.ones((out_channels * 1,), dtype=np.float32)
    downsample_1_eps = 1e-5

def forward(x, in_channels, out_channels, stride=1):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups)
    out = _batch_norm(out, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, bn2_eps)
    identity = _batch_norm(_conv2d(x, downsample_0_weight, downsample_0_bias, downsample_0_stride, downsample_0_padding, downsample_0_dilation, downsample_0_groups), downsample_1_weight, downsample_1_bias, downsample_1_running_mean, downsample_1_running_var, downsample_1_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

