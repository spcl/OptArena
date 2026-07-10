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

def init(in_channels, out_channels, kernel_size, stride, expand_ratio):
    global use_residual, depthwise_conv_0_weight, depthwise_conv_0_bias, depthwise_conv_0_stride, depthwise_conv_0_padding, depthwise_conv_0_dilation, depthwise_conv_0_groups, depthwise_conv_1_weight, depthwise_conv_1_bias, depthwise_conv_1_running_mean, depthwise_conv_1_running_var, depthwise_conv_1_eps, depthwise_conv_2, project_conv_0_weight, project_conv_0_bias, project_conv_0_stride, project_conv_0_padding, project_conv_0_dilation, project_conv_0_groups, project_conv_1_weight, project_conv_1_bias, project_conv_1_running_mean, project_conv_1_running_var, project_conv_1_eps
    use_residual = ((stride == 1) and (in_channels == out_channels))
    depthwise_conv_0_weight = np.zeros((in_channels * expand_ratio, in_channels * expand_ratio // in_channels * expand_ratio) + _as_tuple(kernel_size, 2), dtype=np.float32)
    depthwise_conv_0_bias = np.zeros((in_channels * expand_ratio,), dtype=np.float32)
    depthwise_conv_0_stride = stride
    depthwise_conv_0_padding = (kernel_size - 1) // 2
    depthwise_conv_0_dilation = 1
    depthwise_conv_0_groups = in_channels * expand_ratio
    depthwise_conv_1_weight = np.ones((in_channels * expand_ratio,), dtype=np.float32)
    depthwise_conv_1_bias = np.zeros((in_channels * expand_ratio,), dtype=np.float32)
    depthwise_conv_1_running_mean = np.zeros((in_channels * expand_ratio,), dtype=np.float32)
    depthwise_conv_1_running_var = np.ones((in_channels * expand_ratio,), dtype=np.float32)
    depthwise_conv_1_eps = 1e-5
    depthwise_conv_2 = None
    project_conv_0_weight = np.zeros((out_channels, in_channels * expand_ratio // 1) + _as_tuple(1, 2), dtype=np.float32)
    project_conv_0_bias = np.zeros((out_channels,), dtype=np.float32)
    project_conv_0_stride = 1
    project_conv_0_padding = 0
    project_conv_0_dilation = 1
    project_conv_0_groups = 1
    project_conv_1_weight = np.ones((out_channels,), dtype=np.float32)
    project_conv_1_bias = np.zeros((out_channels,), dtype=np.float32)
    project_conv_1_running_mean = np.zeros((out_channels,), dtype=np.float32)
    project_conv_1_running_var = np.ones((out_channels,), dtype=np.float32)
    project_conv_1_eps = 1e-5

def forward(x, in_channels, out_channels, kernel_size, stride, expand_ratio):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, depthwise_conv_0_weight, depthwise_conv_0_bias, depthwise_conv_0_stride, depthwise_conv_0_padding, depthwise_conv_0_dilation, depthwise_conv_0_groups), depthwise_conv_1_weight, depthwise_conv_1_bias, depthwise_conv_1_running_mean, depthwise_conv_1_running_var, depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, project_conv_0_weight, project_conv_0_bias, project_conv_0_stride, project_conv_0_padding, project_conv_0_dilation, project_conv_0_groups), project_conv_1_weight, project_conv_1_bias, project_conv_1_running_mean, project_conv_1_running_var, project_conv_1_eps)
    if use_residual:
        x += identity
    return x

