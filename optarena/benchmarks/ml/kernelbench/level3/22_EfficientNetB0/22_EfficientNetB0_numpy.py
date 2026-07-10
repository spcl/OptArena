import numpy as np

def _adaptive_avg_pool2d(x, output_size):
    if isinstance(output_size, int): output_size = (output_size, output_size)
    n, c, h, w = x.shape
    out = np.zeros((n, c, output_size[0], output_size[1]), dtype=x.dtype)
    for oy in range(output_size[0]):
        hs = int(np.floor(oy * h / output_size[0]))
        he = int(np.ceil((oy + 1) * h / output_size[0]))
        for ox in range(output_size[1]):
            ws = int(np.floor(ox * w / output_size[1]))
            we = int(np.ceil((ox + 1) * w / output_size[1]))
            out[:, :, oy, ox] = np.mean(x[:, :, hs:he, ws:we], axis=(2, 3))
    return out


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

def _blocks_0_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_0_depthwise_conv_0_weight, blocks_0_depthwise_conv_0_bias, blocks_0_depthwise_conv_0_stride, blocks_0_depthwise_conv_0_padding, blocks_0_depthwise_conv_0_dilation, blocks_0_depthwise_conv_0_groups), blocks_0_depthwise_conv_1_weight, blocks_0_depthwise_conv_1_bias, blocks_0_depthwise_conv_1_running_mean, blocks_0_depthwise_conv_1_running_var, blocks_0_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_0_project_conv_0_weight, blocks_0_project_conv_0_bias, blocks_0_project_conv_0_stride, blocks_0_project_conv_0_padding, blocks_0_project_conv_0_dilation, blocks_0_project_conv_0_groups), blocks_0_project_conv_1_weight, blocks_0_project_conv_1_bias, blocks_0_project_conv_1_running_mean, blocks_0_project_conv_1_running_var, blocks_0_project_conv_1_eps)
    if blocks_0_use_residual:
        x += identity
    return x

def _blocks_1_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_1_depthwise_conv_0_weight, blocks_1_depthwise_conv_0_bias, blocks_1_depthwise_conv_0_stride, blocks_1_depthwise_conv_0_padding, blocks_1_depthwise_conv_0_dilation, blocks_1_depthwise_conv_0_groups), blocks_1_depthwise_conv_1_weight, blocks_1_depthwise_conv_1_bias, blocks_1_depthwise_conv_1_running_mean, blocks_1_depthwise_conv_1_running_var, blocks_1_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_1_project_conv_0_weight, blocks_1_project_conv_0_bias, blocks_1_project_conv_0_stride, blocks_1_project_conv_0_padding, blocks_1_project_conv_0_dilation, blocks_1_project_conv_0_groups), blocks_1_project_conv_1_weight, blocks_1_project_conv_1_bias, blocks_1_project_conv_1_running_mean, blocks_1_project_conv_1_running_var, blocks_1_project_conv_1_eps)
    if blocks_1_use_residual:
        x += identity
    return x

def _blocks_2_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_2_depthwise_conv_0_weight, blocks_2_depthwise_conv_0_bias, blocks_2_depthwise_conv_0_stride, blocks_2_depthwise_conv_0_padding, blocks_2_depthwise_conv_0_dilation, blocks_2_depthwise_conv_0_groups), blocks_2_depthwise_conv_1_weight, blocks_2_depthwise_conv_1_bias, blocks_2_depthwise_conv_1_running_mean, blocks_2_depthwise_conv_1_running_var, blocks_2_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_2_project_conv_0_weight, blocks_2_project_conv_0_bias, blocks_2_project_conv_0_stride, blocks_2_project_conv_0_padding, blocks_2_project_conv_0_dilation, blocks_2_project_conv_0_groups), blocks_2_project_conv_1_weight, blocks_2_project_conv_1_bias, blocks_2_project_conv_1_running_mean, blocks_2_project_conv_1_running_var, blocks_2_project_conv_1_eps)
    if blocks_2_use_residual:
        x += identity
    return x

def _blocks_3_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_3_depthwise_conv_0_weight, blocks_3_depthwise_conv_0_bias, blocks_3_depthwise_conv_0_stride, blocks_3_depthwise_conv_0_padding, blocks_3_depthwise_conv_0_dilation, blocks_3_depthwise_conv_0_groups), blocks_3_depthwise_conv_1_weight, blocks_3_depthwise_conv_1_bias, blocks_3_depthwise_conv_1_running_mean, blocks_3_depthwise_conv_1_running_var, blocks_3_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_3_project_conv_0_weight, blocks_3_project_conv_0_bias, blocks_3_project_conv_0_stride, blocks_3_project_conv_0_padding, blocks_3_project_conv_0_dilation, blocks_3_project_conv_0_groups), blocks_3_project_conv_1_weight, blocks_3_project_conv_1_bias, blocks_3_project_conv_1_running_mean, blocks_3_project_conv_1_running_var, blocks_3_project_conv_1_eps)
    if blocks_3_use_residual:
        x += identity
    return x

def _blocks_4_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_4_depthwise_conv_0_weight, blocks_4_depthwise_conv_0_bias, blocks_4_depthwise_conv_0_stride, blocks_4_depthwise_conv_0_padding, blocks_4_depthwise_conv_0_dilation, blocks_4_depthwise_conv_0_groups), blocks_4_depthwise_conv_1_weight, blocks_4_depthwise_conv_1_bias, blocks_4_depthwise_conv_1_running_mean, blocks_4_depthwise_conv_1_running_var, blocks_4_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_4_project_conv_0_weight, blocks_4_project_conv_0_bias, blocks_4_project_conv_0_stride, blocks_4_project_conv_0_padding, blocks_4_project_conv_0_dilation, blocks_4_project_conv_0_groups), blocks_4_project_conv_1_weight, blocks_4_project_conv_1_bias, blocks_4_project_conv_1_running_mean, blocks_4_project_conv_1_running_var, blocks_4_project_conv_1_eps)
    if blocks_4_use_residual:
        x += identity
    return x

def _blocks_5_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_5_depthwise_conv_0_weight, blocks_5_depthwise_conv_0_bias, blocks_5_depthwise_conv_0_stride, blocks_5_depthwise_conv_0_padding, blocks_5_depthwise_conv_0_dilation, blocks_5_depthwise_conv_0_groups), blocks_5_depthwise_conv_1_weight, blocks_5_depthwise_conv_1_bias, blocks_5_depthwise_conv_1_running_mean, blocks_5_depthwise_conv_1_running_var, blocks_5_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_5_project_conv_0_weight, blocks_5_project_conv_0_bias, blocks_5_project_conv_0_stride, blocks_5_project_conv_0_padding, blocks_5_project_conv_0_dilation, blocks_5_project_conv_0_groups), blocks_5_project_conv_1_weight, blocks_5_project_conv_1_bias, blocks_5_project_conv_1_running_mean, blocks_5_project_conv_1_running_var, blocks_5_project_conv_1_eps)
    if blocks_5_use_residual:
        x += identity
    return x

def _blocks_6_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_6_depthwise_conv_0_weight, blocks_6_depthwise_conv_0_bias, blocks_6_depthwise_conv_0_stride, blocks_6_depthwise_conv_0_padding, blocks_6_depthwise_conv_0_dilation, blocks_6_depthwise_conv_0_groups), blocks_6_depthwise_conv_1_weight, blocks_6_depthwise_conv_1_bias, blocks_6_depthwise_conv_1_running_mean, blocks_6_depthwise_conv_1_running_var, blocks_6_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_6_project_conv_0_weight, blocks_6_project_conv_0_bias, blocks_6_project_conv_0_stride, blocks_6_project_conv_0_padding, blocks_6_project_conv_0_dilation, blocks_6_project_conv_0_groups), blocks_6_project_conv_1_weight, blocks_6_project_conv_1_bias, blocks_6_project_conv_1_running_mean, blocks_6_project_conv_1_running_var, blocks_6_project_conv_1_eps)
    if blocks_6_use_residual:
        x += identity
    return x

def _blocks_7_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_7_depthwise_conv_0_weight, blocks_7_depthwise_conv_0_bias, blocks_7_depthwise_conv_0_stride, blocks_7_depthwise_conv_0_padding, blocks_7_depthwise_conv_0_dilation, blocks_7_depthwise_conv_0_groups), blocks_7_depthwise_conv_1_weight, blocks_7_depthwise_conv_1_bias, blocks_7_depthwise_conv_1_running_mean, blocks_7_depthwise_conv_1_running_var, blocks_7_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_7_project_conv_0_weight, blocks_7_project_conv_0_bias, blocks_7_project_conv_0_stride, blocks_7_project_conv_0_padding, blocks_7_project_conv_0_dilation, blocks_7_project_conv_0_groups), blocks_7_project_conv_1_weight, blocks_7_project_conv_1_bias, blocks_7_project_conv_1_running_mean, blocks_7_project_conv_1_running_var, blocks_7_project_conv_1_eps)
    if blocks_7_use_residual:
        x += identity
    return x

def _blocks_8_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_8_depthwise_conv_0_weight, blocks_8_depthwise_conv_0_bias, blocks_8_depthwise_conv_0_stride, blocks_8_depthwise_conv_0_padding, blocks_8_depthwise_conv_0_dilation, blocks_8_depthwise_conv_0_groups), blocks_8_depthwise_conv_1_weight, blocks_8_depthwise_conv_1_bias, blocks_8_depthwise_conv_1_running_mean, blocks_8_depthwise_conv_1_running_var, blocks_8_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_8_project_conv_0_weight, blocks_8_project_conv_0_bias, blocks_8_project_conv_0_stride, blocks_8_project_conv_0_padding, blocks_8_project_conv_0_dilation, blocks_8_project_conv_0_groups), blocks_8_project_conv_1_weight, blocks_8_project_conv_1_bias, blocks_8_project_conv_1_running_mean, blocks_8_project_conv_1_running_var, blocks_8_project_conv_1_eps)
    if blocks_8_use_residual:
        x += identity
    return x

def _blocks_9_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_9_depthwise_conv_0_weight, blocks_9_depthwise_conv_0_bias, blocks_9_depthwise_conv_0_stride, blocks_9_depthwise_conv_0_padding, blocks_9_depthwise_conv_0_dilation, blocks_9_depthwise_conv_0_groups), blocks_9_depthwise_conv_1_weight, blocks_9_depthwise_conv_1_bias, blocks_9_depthwise_conv_1_running_mean, blocks_9_depthwise_conv_1_running_var, blocks_9_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_9_project_conv_0_weight, blocks_9_project_conv_0_bias, blocks_9_project_conv_0_stride, blocks_9_project_conv_0_padding, blocks_9_project_conv_0_dilation, blocks_9_project_conv_0_groups), blocks_9_project_conv_1_weight, blocks_9_project_conv_1_bias, blocks_9_project_conv_1_running_mean, blocks_9_project_conv_1_running_var, blocks_9_project_conv_1_eps)
    if blocks_9_use_residual:
        x += identity
    return x

def _blocks_10_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_10_depthwise_conv_0_weight, blocks_10_depthwise_conv_0_bias, blocks_10_depthwise_conv_0_stride, blocks_10_depthwise_conv_0_padding, blocks_10_depthwise_conv_0_dilation, blocks_10_depthwise_conv_0_groups), blocks_10_depthwise_conv_1_weight, blocks_10_depthwise_conv_1_bias, blocks_10_depthwise_conv_1_running_mean, blocks_10_depthwise_conv_1_running_var, blocks_10_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_10_project_conv_0_weight, blocks_10_project_conv_0_bias, blocks_10_project_conv_0_stride, blocks_10_project_conv_0_padding, blocks_10_project_conv_0_dilation, blocks_10_project_conv_0_groups), blocks_10_project_conv_1_weight, blocks_10_project_conv_1_bias, blocks_10_project_conv_1_running_mean, blocks_10_project_conv_1_running_var, blocks_10_project_conv_1_eps)
    if blocks_10_use_residual:
        x += identity
    return x

def _blocks_11_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_11_depthwise_conv_0_weight, blocks_11_depthwise_conv_0_bias, blocks_11_depthwise_conv_0_stride, blocks_11_depthwise_conv_0_padding, blocks_11_depthwise_conv_0_dilation, blocks_11_depthwise_conv_0_groups), blocks_11_depthwise_conv_1_weight, blocks_11_depthwise_conv_1_bias, blocks_11_depthwise_conv_1_running_mean, blocks_11_depthwise_conv_1_running_var, blocks_11_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_11_project_conv_0_weight, blocks_11_project_conv_0_bias, blocks_11_project_conv_0_stride, blocks_11_project_conv_0_padding, blocks_11_project_conv_0_dilation, blocks_11_project_conv_0_groups), blocks_11_project_conv_1_weight, blocks_11_project_conv_1_bias, blocks_11_project_conv_1_running_mean, blocks_11_project_conv_1_running_var, blocks_11_project_conv_1_eps)
    if blocks_11_use_residual:
        x += identity
    return x

def _blocks_12_forward(x):
    identity = x
    x = np.clip(_batch_norm(_conv2d(x, blocks_12_depthwise_conv_0_weight, blocks_12_depthwise_conv_0_bias, blocks_12_depthwise_conv_0_stride, blocks_12_depthwise_conv_0_padding, blocks_12_depthwise_conv_0_dilation, blocks_12_depthwise_conv_0_groups), blocks_12_depthwise_conv_1_weight, blocks_12_depthwise_conv_1_bias, blocks_12_depthwise_conv_1_running_mean, blocks_12_depthwise_conv_1_running_var, blocks_12_depthwise_conv_1_eps), 0.0, 6.0)
    x = _batch_norm(_conv2d(x, blocks_12_project_conv_0_weight, blocks_12_project_conv_0_bias, blocks_12_project_conv_0_stride, blocks_12_project_conv_0_padding, blocks_12_project_conv_0_dilation, blocks_12_project_conv_0_groups), blocks_12_project_conv_1_weight, blocks_12_project_conv_1_bias, blocks_12_project_conv_1_running_mean, blocks_12_project_conv_1_running_var, blocks_12_project_conv_1_eps)
    if blocks_12_use_residual:
        x += identity
    return x

def init(num_classes=1000):
    global conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps, blocks_0_use_residual, blocks_0_depthwise_conv_0_weight, blocks_0_depthwise_conv_0_bias, blocks_0_depthwise_conv_0_stride, blocks_0_depthwise_conv_0_padding, blocks_0_depthwise_conv_0_dilation, blocks_0_depthwise_conv_0_groups, blocks_0_depthwise_conv_1_weight, blocks_0_depthwise_conv_1_bias, blocks_0_depthwise_conv_1_running_mean, blocks_0_depthwise_conv_1_running_var, blocks_0_depthwise_conv_1_eps, blocks_0_depthwise_conv_2, blocks_0_project_conv_0_weight, blocks_0_project_conv_0_bias, blocks_0_project_conv_0_stride, blocks_0_project_conv_0_padding, blocks_0_project_conv_0_dilation, blocks_0_project_conv_0_groups, blocks_0_project_conv_1_weight, blocks_0_project_conv_1_bias, blocks_0_project_conv_1_running_mean, blocks_0_project_conv_1_running_var, blocks_0_project_conv_1_eps, blocks_1_use_residual, blocks_1_depthwise_conv_0_weight, blocks_1_depthwise_conv_0_bias, blocks_1_depthwise_conv_0_stride, blocks_1_depthwise_conv_0_padding, blocks_1_depthwise_conv_0_dilation, blocks_1_depthwise_conv_0_groups, blocks_1_depthwise_conv_1_weight, blocks_1_depthwise_conv_1_bias, blocks_1_depthwise_conv_1_running_mean, blocks_1_depthwise_conv_1_running_var, blocks_1_depthwise_conv_1_eps, blocks_1_depthwise_conv_2, blocks_1_project_conv_0_weight, blocks_1_project_conv_0_bias, blocks_1_project_conv_0_stride, blocks_1_project_conv_0_padding, blocks_1_project_conv_0_dilation, blocks_1_project_conv_0_groups, blocks_1_project_conv_1_weight, blocks_1_project_conv_1_bias, blocks_1_project_conv_1_running_mean, blocks_1_project_conv_1_running_var, blocks_1_project_conv_1_eps, blocks_2_use_residual, blocks_2_depthwise_conv_0_weight, blocks_2_depthwise_conv_0_bias, blocks_2_depthwise_conv_0_stride, blocks_2_depthwise_conv_0_padding, blocks_2_depthwise_conv_0_dilation, blocks_2_depthwise_conv_0_groups, blocks_2_depthwise_conv_1_weight, blocks_2_depthwise_conv_1_bias, blocks_2_depthwise_conv_1_running_mean, blocks_2_depthwise_conv_1_running_var, blocks_2_depthwise_conv_1_eps, blocks_2_depthwise_conv_2, blocks_2_project_conv_0_weight, blocks_2_project_conv_0_bias, blocks_2_project_conv_0_stride, blocks_2_project_conv_0_padding, blocks_2_project_conv_0_dilation, blocks_2_project_conv_0_groups, blocks_2_project_conv_1_weight, blocks_2_project_conv_1_bias, blocks_2_project_conv_1_running_mean, blocks_2_project_conv_1_running_var, blocks_2_project_conv_1_eps, blocks_3_use_residual, blocks_3_depthwise_conv_0_weight, blocks_3_depthwise_conv_0_bias, blocks_3_depthwise_conv_0_stride, blocks_3_depthwise_conv_0_padding, blocks_3_depthwise_conv_0_dilation, blocks_3_depthwise_conv_0_groups, blocks_3_depthwise_conv_1_weight, blocks_3_depthwise_conv_1_bias, blocks_3_depthwise_conv_1_running_mean, blocks_3_depthwise_conv_1_running_var, blocks_3_depthwise_conv_1_eps, blocks_3_depthwise_conv_2, blocks_3_project_conv_0_weight, blocks_3_project_conv_0_bias, blocks_3_project_conv_0_stride, blocks_3_project_conv_0_padding, blocks_3_project_conv_0_dilation, blocks_3_project_conv_0_groups, blocks_3_project_conv_1_weight, blocks_3_project_conv_1_bias, blocks_3_project_conv_1_running_mean, blocks_3_project_conv_1_running_var, blocks_3_project_conv_1_eps, blocks_4_use_residual, blocks_4_depthwise_conv_0_weight, blocks_4_depthwise_conv_0_bias, blocks_4_depthwise_conv_0_stride, blocks_4_depthwise_conv_0_padding, blocks_4_depthwise_conv_0_dilation, blocks_4_depthwise_conv_0_groups, blocks_4_depthwise_conv_1_weight, blocks_4_depthwise_conv_1_bias, blocks_4_depthwise_conv_1_running_mean, blocks_4_depthwise_conv_1_running_var, blocks_4_depthwise_conv_1_eps, blocks_4_depthwise_conv_2, blocks_4_project_conv_0_weight, blocks_4_project_conv_0_bias, blocks_4_project_conv_0_stride, blocks_4_project_conv_0_padding, blocks_4_project_conv_0_dilation, blocks_4_project_conv_0_groups, blocks_4_project_conv_1_weight, blocks_4_project_conv_1_bias, blocks_4_project_conv_1_running_mean, blocks_4_project_conv_1_running_var, blocks_4_project_conv_1_eps, blocks_5_use_residual, blocks_5_depthwise_conv_0_weight, blocks_5_depthwise_conv_0_bias, blocks_5_depthwise_conv_0_stride, blocks_5_depthwise_conv_0_padding, blocks_5_depthwise_conv_0_dilation, blocks_5_depthwise_conv_0_groups, blocks_5_depthwise_conv_1_weight, blocks_5_depthwise_conv_1_bias, blocks_5_depthwise_conv_1_running_mean, blocks_5_depthwise_conv_1_running_var, blocks_5_depthwise_conv_1_eps, blocks_5_depthwise_conv_2, blocks_5_project_conv_0_weight, blocks_5_project_conv_0_bias, blocks_5_project_conv_0_stride, blocks_5_project_conv_0_padding, blocks_5_project_conv_0_dilation, blocks_5_project_conv_0_groups, blocks_5_project_conv_1_weight, blocks_5_project_conv_1_bias, blocks_5_project_conv_1_running_mean, blocks_5_project_conv_1_running_var, blocks_5_project_conv_1_eps, blocks_6_use_residual, blocks_6_depthwise_conv_0_weight, blocks_6_depthwise_conv_0_bias, blocks_6_depthwise_conv_0_stride, blocks_6_depthwise_conv_0_padding, blocks_6_depthwise_conv_0_dilation, blocks_6_depthwise_conv_0_groups, blocks_6_depthwise_conv_1_weight, blocks_6_depthwise_conv_1_bias, blocks_6_depthwise_conv_1_running_mean, blocks_6_depthwise_conv_1_running_var, blocks_6_depthwise_conv_1_eps, blocks_6_depthwise_conv_2, blocks_6_project_conv_0_weight, blocks_6_project_conv_0_bias, blocks_6_project_conv_0_stride, blocks_6_project_conv_0_padding, blocks_6_project_conv_0_dilation, blocks_6_project_conv_0_groups, blocks_6_project_conv_1_weight, blocks_6_project_conv_1_bias, blocks_6_project_conv_1_running_mean, blocks_6_project_conv_1_running_var, blocks_6_project_conv_1_eps, blocks_7_use_residual, blocks_7_depthwise_conv_0_weight, blocks_7_depthwise_conv_0_bias, blocks_7_depthwise_conv_0_stride, blocks_7_depthwise_conv_0_padding, blocks_7_depthwise_conv_0_dilation, blocks_7_depthwise_conv_0_groups, blocks_7_depthwise_conv_1_weight, blocks_7_depthwise_conv_1_bias, blocks_7_depthwise_conv_1_running_mean, blocks_7_depthwise_conv_1_running_var, blocks_7_depthwise_conv_1_eps, blocks_7_depthwise_conv_2, blocks_7_project_conv_0_weight, blocks_7_project_conv_0_bias, blocks_7_project_conv_0_stride, blocks_7_project_conv_0_padding, blocks_7_project_conv_0_dilation, blocks_7_project_conv_0_groups, blocks_7_project_conv_1_weight, blocks_7_project_conv_1_bias, blocks_7_project_conv_1_running_mean, blocks_7_project_conv_1_running_var, blocks_7_project_conv_1_eps, blocks_8_use_residual, blocks_8_depthwise_conv_0_weight, blocks_8_depthwise_conv_0_bias, blocks_8_depthwise_conv_0_stride, blocks_8_depthwise_conv_0_padding, blocks_8_depthwise_conv_0_dilation, blocks_8_depthwise_conv_0_groups, blocks_8_depthwise_conv_1_weight, blocks_8_depthwise_conv_1_bias, blocks_8_depthwise_conv_1_running_mean, blocks_8_depthwise_conv_1_running_var, blocks_8_depthwise_conv_1_eps, blocks_8_depthwise_conv_2, blocks_8_project_conv_0_weight, blocks_8_project_conv_0_bias, blocks_8_project_conv_0_stride, blocks_8_project_conv_0_padding, blocks_8_project_conv_0_dilation, blocks_8_project_conv_0_groups, blocks_8_project_conv_1_weight, blocks_8_project_conv_1_bias, blocks_8_project_conv_1_running_mean, blocks_8_project_conv_1_running_var, blocks_8_project_conv_1_eps, blocks_9_use_residual, blocks_9_depthwise_conv_0_weight, blocks_9_depthwise_conv_0_bias, blocks_9_depthwise_conv_0_stride, blocks_9_depthwise_conv_0_padding, blocks_9_depthwise_conv_0_dilation, blocks_9_depthwise_conv_0_groups, blocks_9_depthwise_conv_1_weight, blocks_9_depthwise_conv_1_bias, blocks_9_depthwise_conv_1_running_mean, blocks_9_depthwise_conv_1_running_var, blocks_9_depthwise_conv_1_eps, blocks_9_depthwise_conv_2, blocks_9_project_conv_0_weight, blocks_9_project_conv_0_bias, blocks_9_project_conv_0_stride, blocks_9_project_conv_0_padding, blocks_9_project_conv_0_dilation, blocks_9_project_conv_0_groups, blocks_9_project_conv_1_weight, blocks_9_project_conv_1_bias, blocks_9_project_conv_1_running_mean, blocks_9_project_conv_1_running_var, blocks_9_project_conv_1_eps, blocks_10_use_residual, blocks_10_depthwise_conv_0_weight, blocks_10_depthwise_conv_0_bias, blocks_10_depthwise_conv_0_stride, blocks_10_depthwise_conv_0_padding, blocks_10_depthwise_conv_0_dilation, blocks_10_depthwise_conv_0_groups, blocks_10_depthwise_conv_1_weight, blocks_10_depthwise_conv_1_bias, blocks_10_depthwise_conv_1_running_mean, blocks_10_depthwise_conv_1_running_var, blocks_10_depthwise_conv_1_eps, blocks_10_depthwise_conv_2, blocks_10_project_conv_0_weight, blocks_10_project_conv_0_bias, blocks_10_project_conv_0_stride, blocks_10_project_conv_0_padding, blocks_10_project_conv_0_dilation, blocks_10_project_conv_0_groups, blocks_10_project_conv_1_weight, blocks_10_project_conv_1_bias, blocks_10_project_conv_1_running_mean, blocks_10_project_conv_1_running_var, blocks_10_project_conv_1_eps, blocks_11_use_residual, blocks_11_depthwise_conv_0_weight, blocks_11_depthwise_conv_0_bias, blocks_11_depthwise_conv_0_stride, blocks_11_depthwise_conv_0_padding, blocks_11_depthwise_conv_0_dilation, blocks_11_depthwise_conv_0_groups, blocks_11_depthwise_conv_1_weight, blocks_11_depthwise_conv_1_bias, blocks_11_depthwise_conv_1_running_mean, blocks_11_depthwise_conv_1_running_var, blocks_11_depthwise_conv_1_eps, blocks_11_depthwise_conv_2, blocks_11_project_conv_0_weight, blocks_11_project_conv_0_bias, blocks_11_project_conv_0_stride, blocks_11_project_conv_0_padding, blocks_11_project_conv_0_dilation, blocks_11_project_conv_0_groups, blocks_11_project_conv_1_weight, blocks_11_project_conv_1_bias, blocks_11_project_conv_1_running_mean, blocks_11_project_conv_1_running_var, blocks_11_project_conv_1_eps, blocks_12_use_residual, blocks_12_depthwise_conv_0_weight, blocks_12_depthwise_conv_0_bias, blocks_12_depthwise_conv_0_stride, blocks_12_depthwise_conv_0_padding, blocks_12_depthwise_conv_0_dilation, blocks_12_depthwise_conv_0_groups, blocks_12_depthwise_conv_1_weight, blocks_12_depthwise_conv_1_bias, blocks_12_depthwise_conv_1_running_mean, blocks_12_depthwise_conv_1_running_var, blocks_12_depthwise_conv_1_eps, blocks_12_depthwise_conv_2, blocks_12_project_conv_0_weight, blocks_12_project_conv_0_bias, blocks_12_project_conv_0_stride, blocks_12_project_conv_0_padding, blocks_12_project_conv_0_dilation, blocks_12_project_conv_0_groups, blocks_12_project_conv_1_weight, blocks_12_project_conv_1_bias, blocks_12_project_conv_1_running_mean, blocks_12_project_conv_1_running_var, blocks_12_project_conv_1_eps, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, bn2_eps, fc_weight, fc_bias
    conv1_weight = np.zeros((32, 3 // 1) + _as_tuple(3, 2), dtype=np.float32)
    conv1_bias = np.zeros((32,), dtype=np.float32)
    conv1_stride = 2
    conv1_padding = 1
    conv1_dilation = 1
    conv1_groups = 1
    bn1_weight = np.ones((32,), dtype=np.float32)
    bn1_bias = np.zeros((32,), dtype=np.float32)
    bn1_running_mean = np.zeros((32,), dtype=np.float32)
    bn1_running_var = np.ones((32,), dtype=np.float32)
    bn1_eps = 1e-5
    blocks_0_use_residual = ((1 == 1) and (32 == 16))
    blocks_0_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_0_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_0_depthwise_conv_0_stride = 1
    blocks_0_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_0_depthwise_conv_0_dilation = 1
    blocks_0_depthwise_conv_0_groups = 32 * 1
    blocks_0_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_0_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_0_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_0_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_0_depthwise_conv_1_eps = 1e-5
    blocks_0_depthwise_conv_2 = None
    blocks_0_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_0_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_0_project_conv_0_stride = 1
    blocks_0_project_conv_0_padding = 0
    blocks_0_project_conv_0_dilation = 1
    blocks_0_project_conv_0_groups = 1
    blocks_0_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_0_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_0_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_0_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_0_project_conv_1_eps = 1e-5
    blocks_1_use_residual = ((1 == 1) and (32 == 16))
    blocks_1_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_1_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_1_depthwise_conv_0_stride = 1
    blocks_1_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_1_depthwise_conv_0_dilation = 1
    blocks_1_depthwise_conv_0_groups = 32 * 1
    blocks_1_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_1_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_1_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_1_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_1_depthwise_conv_1_eps = 1e-5
    blocks_1_depthwise_conv_2 = None
    blocks_1_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_1_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_1_project_conv_0_stride = 1
    blocks_1_project_conv_0_padding = 0
    blocks_1_project_conv_0_dilation = 1
    blocks_1_project_conv_0_groups = 1
    blocks_1_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_1_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_1_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_1_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_1_project_conv_1_eps = 1e-5
    blocks_2_use_residual = ((1 == 1) and (32 == 16))
    blocks_2_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_2_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_2_depthwise_conv_0_stride = 1
    blocks_2_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_2_depthwise_conv_0_dilation = 1
    blocks_2_depthwise_conv_0_groups = 32 * 1
    blocks_2_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_2_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_2_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_2_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_2_depthwise_conv_1_eps = 1e-5
    blocks_2_depthwise_conv_2 = None
    blocks_2_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_2_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_2_project_conv_0_stride = 1
    blocks_2_project_conv_0_padding = 0
    blocks_2_project_conv_0_dilation = 1
    blocks_2_project_conv_0_groups = 1
    blocks_2_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_2_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_2_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_2_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_2_project_conv_1_eps = 1e-5
    blocks_3_use_residual = ((1 == 1) and (32 == 16))
    blocks_3_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_3_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_3_depthwise_conv_0_stride = 1
    blocks_3_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_3_depthwise_conv_0_dilation = 1
    blocks_3_depthwise_conv_0_groups = 32 * 1
    blocks_3_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_3_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_3_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_3_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_3_depthwise_conv_1_eps = 1e-5
    blocks_3_depthwise_conv_2 = None
    blocks_3_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_3_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_3_project_conv_0_stride = 1
    blocks_3_project_conv_0_padding = 0
    blocks_3_project_conv_0_dilation = 1
    blocks_3_project_conv_0_groups = 1
    blocks_3_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_3_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_3_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_3_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_3_project_conv_1_eps = 1e-5
    blocks_4_use_residual = ((1 == 1) and (32 == 16))
    blocks_4_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_4_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_4_depthwise_conv_0_stride = 1
    blocks_4_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_4_depthwise_conv_0_dilation = 1
    blocks_4_depthwise_conv_0_groups = 32 * 1
    blocks_4_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_4_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_4_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_4_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_4_depthwise_conv_1_eps = 1e-5
    blocks_4_depthwise_conv_2 = None
    blocks_4_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_4_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_4_project_conv_0_stride = 1
    blocks_4_project_conv_0_padding = 0
    blocks_4_project_conv_0_dilation = 1
    blocks_4_project_conv_0_groups = 1
    blocks_4_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_4_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_4_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_4_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_4_project_conv_1_eps = 1e-5
    blocks_5_use_residual = ((1 == 1) and (32 == 16))
    blocks_5_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_5_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_5_depthwise_conv_0_stride = 1
    blocks_5_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_5_depthwise_conv_0_dilation = 1
    blocks_5_depthwise_conv_0_groups = 32 * 1
    blocks_5_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_5_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_5_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_5_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_5_depthwise_conv_1_eps = 1e-5
    blocks_5_depthwise_conv_2 = None
    blocks_5_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_5_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_5_project_conv_0_stride = 1
    blocks_5_project_conv_0_padding = 0
    blocks_5_project_conv_0_dilation = 1
    blocks_5_project_conv_0_groups = 1
    blocks_5_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_5_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_5_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_5_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_5_project_conv_1_eps = 1e-5
    blocks_6_use_residual = ((1 == 1) and (32 == 16))
    blocks_6_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_6_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_6_depthwise_conv_0_stride = 1
    blocks_6_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_6_depthwise_conv_0_dilation = 1
    blocks_6_depthwise_conv_0_groups = 32 * 1
    blocks_6_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_6_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_6_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_6_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_6_depthwise_conv_1_eps = 1e-5
    blocks_6_depthwise_conv_2 = None
    blocks_6_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_6_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_6_project_conv_0_stride = 1
    blocks_6_project_conv_0_padding = 0
    blocks_6_project_conv_0_dilation = 1
    blocks_6_project_conv_0_groups = 1
    blocks_6_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_6_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_6_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_6_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_6_project_conv_1_eps = 1e-5
    blocks_7_use_residual = ((1 == 1) and (32 == 16))
    blocks_7_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_7_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_7_depthwise_conv_0_stride = 1
    blocks_7_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_7_depthwise_conv_0_dilation = 1
    blocks_7_depthwise_conv_0_groups = 32 * 1
    blocks_7_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_7_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_7_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_7_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_7_depthwise_conv_1_eps = 1e-5
    blocks_7_depthwise_conv_2 = None
    blocks_7_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_7_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_7_project_conv_0_stride = 1
    blocks_7_project_conv_0_padding = 0
    blocks_7_project_conv_0_dilation = 1
    blocks_7_project_conv_0_groups = 1
    blocks_7_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_7_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_7_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_7_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_7_project_conv_1_eps = 1e-5
    blocks_8_use_residual = ((1 == 1) and (32 == 16))
    blocks_8_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_8_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_8_depthwise_conv_0_stride = 1
    blocks_8_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_8_depthwise_conv_0_dilation = 1
    blocks_8_depthwise_conv_0_groups = 32 * 1
    blocks_8_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_8_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_8_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_8_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_8_depthwise_conv_1_eps = 1e-5
    blocks_8_depthwise_conv_2 = None
    blocks_8_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_8_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_8_project_conv_0_stride = 1
    blocks_8_project_conv_0_padding = 0
    blocks_8_project_conv_0_dilation = 1
    blocks_8_project_conv_0_groups = 1
    blocks_8_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_8_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_8_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_8_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_8_project_conv_1_eps = 1e-5
    blocks_9_use_residual = ((1 == 1) and (32 == 16))
    blocks_9_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_9_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_9_depthwise_conv_0_stride = 1
    blocks_9_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_9_depthwise_conv_0_dilation = 1
    blocks_9_depthwise_conv_0_groups = 32 * 1
    blocks_9_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_9_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_9_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_9_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_9_depthwise_conv_1_eps = 1e-5
    blocks_9_depthwise_conv_2 = None
    blocks_9_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_9_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_9_project_conv_0_stride = 1
    blocks_9_project_conv_0_padding = 0
    blocks_9_project_conv_0_dilation = 1
    blocks_9_project_conv_0_groups = 1
    blocks_9_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_9_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_9_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_9_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_9_project_conv_1_eps = 1e-5
    blocks_10_use_residual = ((1 == 1) and (32 == 16))
    blocks_10_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_10_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_10_depthwise_conv_0_stride = 1
    blocks_10_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_10_depthwise_conv_0_dilation = 1
    blocks_10_depthwise_conv_0_groups = 32 * 1
    blocks_10_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_10_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_10_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_10_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_10_depthwise_conv_1_eps = 1e-5
    blocks_10_depthwise_conv_2 = None
    blocks_10_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_10_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_10_project_conv_0_stride = 1
    blocks_10_project_conv_0_padding = 0
    blocks_10_project_conv_0_dilation = 1
    blocks_10_project_conv_0_groups = 1
    blocks_10_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_10_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_10_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_10_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_10_project_conv_1_eps = 1e-5
    blocks_11_use_residual = ((1 == 1) and (32 == 16))
    blocks_11_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_11_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_11_depthwise_conv_0_stride = 1
    blocks_11_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_11_depthwise_conv_0_dilation = 1
    blocks_11_depthwise_conv_0_groups = 32 * 1
    blocks_11_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_11_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_11_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_11_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_11_depthwise_conv_1_eps = 1e-5
    blocks_11_depthwise_conv_2 = None
    blocks_11_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_11_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_11_project_conv_0_stride = 1
    blocks_11_project_conv_0_padding = 0
    blocks_11_project_conv_0_dilation = 1
    blocks_11_project_conv_0_groups = 1
    blocks_11_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_11_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_11_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_11_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_11_project_conv_1_eps = 1e-5
    blocks_12_use_residual = ((1 == 1) and (32 == 16))
    blocks_12_depthwise_conv_0_weight = np.zeros((32 * 1, 32 * 1 // 32 * 1) + _as_tuple(3, 2), dtype=np.float32)
    blocks_12_depthwise_conv_0_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_12_depthwise_conv_0_stride = 1
    blocks_12_depthwise_conv_0_padding = (3 - 1) // 2
    blocks_12_depthwise_conv_0_dilation = 1
    blocks_12_depthwise_conv_0_groups = 32 * 1
    blocks_12_depthwise_conv_1_weight = np.ones((32 * 1,), dtype=np.float32)
    blocks_12_depthwise_conv_1_bias = np.zeros((32 * 1,), dtype=np.float32)
    blocks_12_depthwise_conv_1_running_mean = np.zeros((32 * 1,), dtype=np.float32)
    blocks_12_depthwise_conv_1_running_var = np.ones((32 * 1,), dtype=np.float32)
    blocks_12_depthwise_conv_1_eps = 1e-5
    blocks_12_depthwise_conv_2 = None
    blocks_12_project_conv_0_weight = np.zeros((16, 32 * 1 // 1) + _as_tuple(1, 2), dtype=np.float32)
    blocks_12_project_conv_0_bias = np.zeros((16,), dtype=np.float32)
    blocks_12_project_conv_0_stride = 1
    blocks_12_project_conv_0_padding = 0
    blocks_12_project_conv_0_dilation = 1
    blocks_12_project_conv_0_groups = 1
    blocks_12_project_conv_1_weight = np.ones((16,), dtype=np.float32)
    blocks_12_project_conv_1_bias = np.zeros((16,), dtype=np.float32)
    blocks_12_project_conv_1_running_mean = np.zeros((16,), dtype=np.float32)
    blocks_12_project_conv_1_running_var = np.ones((16,), dtype=np.float32)
    blocks_12_project_conv_1_eps = 1e-5
    conv2_weight = np.zeros((1280, 320 // 1) + _as_tuple(1, 2), dtype=np.float32)
    conv2_bias = np.zeros((1280,), dtype=np.float32)
    conv2_stride = 1
    conv2_padding = 0
    conv2_dilation = 1
    conv2_groups = 1
    bn2_weight = np.ones((1280,), dtype=np.float32)
    bn2_bias = np.zeros((1280,), dtype=np.float32)
    bn2_running_mean = np.zeros((1280,), dtype=np.float32)
    bn2_running_var = np.ones((1280,), dtype=np.float32)
    bn2_eps = 1e-5
    fc_weight = np.zeros((num_classes, 1280), dtype=np.float32)
    fc_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000):
    x = np.maximum(_batch_norm(_conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups), bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps), 0)
    x = _blocks_12_forward(_blocks_11_forward(_blocks_10_forward(_blocks_9_forward(_blocks_8_forward(_blocks_7_forward(_blocks_6_forward(_blocks_5_forward(_blocks_4_forward(_blocks_3_forward(_blocks_2_forward(_blocks_1_forward(_blocks_0_forward(x)))))))))))))
    x = np.maximum(_batch_norm(_conv2d(x, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups), bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, bn2_eps), 0)
    x = _adaptive_avg_pool2d(x, (1, 1))
    x = np.reshape(x, (x.shape[0], (-1)))
    x = ((x) @ fc_weight.T + fc_bias)
    return x

