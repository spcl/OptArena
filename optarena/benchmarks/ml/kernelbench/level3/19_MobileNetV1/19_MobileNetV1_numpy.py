import numpy as np

def _as_tuple(value, dims):
    if isinstance(value, tuple):
        return value
    return tuple(value for _ in range(dims))

def _avgpool2d(x, kernel_size, stride, padding):
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size,)
    if stride is None: stride = kernel_size
    if isinstance(stride, int): stride = (stride, stride,)
    if isinstance(padding, int): padding = (padding, padding,)
    padded_shape = (x.shape[0], x.shape[1]) + tuple(x.shape[i + 2] + 2 * padding[i] for i in range(2))
    fill = -np.inf if "mean" == "max" else 0.0
    padded = np.full(padded_shape, fill, dtype=x.dtype)
    src = tuple(slice(padding[i], padding[i] + x.shape[i + 2]) for i in range(2))
    padded[(slice(None), slice(None)) + src] = x
    out_shape = tuple((padded_shape[i + 2] - kernel_size[i]) // stride[i] + 1 for i in range(2))
    out = np.zeros((x.shape[0], x.shape[1]) + out_shape, dtype=x.dtype)
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            for oy in range(out_shape[0]):
                for ox in range(out_shape[1]):
                    sy = oy * stride[0]
                    sx = ox * stride[1]
                    window = padded[(b, c, slice(sy, sy + kernel_size[0]), slice(sx, sx + kernel_size[1]))]
                    out[b, c, oy, ox] = np.mean(window)
    return out


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

def init(num_classes=1000, input_channels=3, alpha=1.0):
    global model_0_0_weight, model_0_0_bias, model_0_0_stride, model_0_0_padding, model_0_0_dilation, model_0_0_groups, model_0_1_weight, model_0_1_bias, model_0_1_running_mean, model_0_1_running_var, model_0_1_eps, model_0_2, model_1_0_weight, model_1_0_bias, model_1_0_stride, model_1_0_padding, model_1_0_dilation, model_1_0_groups, model_1_1_weight, model_1_1_bias, model_1_1_running_mean, model_1_1_running_var, model_1_1_eps, model_1_2, model_1_3_weight, model_1_3_bias, model_1_3_stride, model_1_3_padding, model_1_3_dilation, model_1_3_groups, model_1_4_weight, model_1_4_bias, model_1_4_running_mean, model_1_4_running_var, model_1_4_eps, model_1_5, model_2_0_weight, model_2_0_bias, model_2_0_stride, model_2_0_padding, model_2_0_dilation, model_2_0_groups, model_2_1_weight, model_2_1_bias, model_2_1_running_mean, model_2_1_running_var, model_2_1_eps, model_2_2, model_2_3_weight, model_2_3_bias, model_2_3_stride, model_2_3_padding, model_2_3_dilation, model_2_3_groups, model_2_4_weight, model_2_4_bias, model_2_4_running_mean, model_2_4_running_var, model_2_4_eps, model_2_5, model_3_0_weight, model_3_0_bias, model_3_0_stride, model_3_0_padding, model_3_0_dilation, model_3_0_groups, model_3_1_weight, model_3_1_bias, model_3_1_running_mean, model_3_1_running_var, model_3_1_eps, model_3_2, model_3_3_weight, model_3_3_bias, model_3_3_stride, model_3_3_padding, model_3_3_dilation, model_3_3_groups, model_3_4_weight, model_3_4_bias, model_3_4_running_mean, model_3_4_running_var, model_3_4_eps, model_3_5, model_4_0_weight, model_4_0_bias, model_4_0_stride, model_4_0_padding, model_4_0_dilation, model_4_0_groups, model_4_1_weight, model_4_1_bias, model_4_1_running_mean, model_4_1_running_var, model_4_1_eps, model_4_2, model_4_3_weight, model_4_3_bias, model_4_3_stride, model_4_3_padding, model_4_3_dilation, model_4_3_groups, model_4_4_weight, model_4_4_bias, model_4_4_running_mean, model_4_4_running_var, model_4_4_eps, model_4_5, model_5_0_weight, model_5_0_bias, model_5_0_stride, model_5_0_padding, model_5_0_dilation, model_5_0_groups, model_5_1_weight, model_5_1_bias, model_5_1_running_mean, model_5_1_running_var, model_5_1_eps, model_5_2, model_5_3_weight, model_5_3_bias, model_5_3_stride, model_5_3_padding, model_5_3_dilation, model_5_3_groups, model_5_4_weight, model_5_4_bias, model_5_4_running_mean, model_5_4_running_var, model_5_4_eps, model_5_5, model_6_0_weight, model_6_0_bias, model_6_0_stride, model_6_0_padding, model_6_0_dilation, model_6_0_groups, model_6_1_weight, model_6_1_bias, model_6_1_running_mean, model_6_1_running_var, model_6_1_eps, model_6_2, model_6_3_weight, model_6_3_bias, model_6_3_stride, model_6_3_padding, model_6_3_dilation, model_6_3_groups, model_6_4_weight, model_6_4_bias, model_6_4_running_mean, model_6_4_running_var, model_6_4_eps, model_6_5, model_7_0_weight, model_7_0_bias, model_7_0_stride, model_7_0_padding, model_7_0_dilation, model_7_0_groups, model_7_1_weight, model_7_1_bias, model_7_1_running_mean, model_7_1_running_var, model_7_1_eps, model_7_2, model_7_3_weight, model_7_3_bias, model_7_3_stride, model_7_3_padding, model_7_3_dilation, model_7_3_groups, model_7_4_weight, model_7_4_bias, model_7_4_running_mean, model_7_4_running_var, model_7_4_eps, model_7_5, model_8_0_weight, model_8_0_bias, model_8_0_stride, model_8_0_padding, model_8_0_dilation, model_8_0_groups, model_8_1_weight, model_8_1_bias, model_8_1_running_mean, model_8_1_running_var, model_8_1_eps, model_8_2, model_8_3_weight, model_8_3_bias, model_8_3_stride, model_8_3_padding, model_8_3_dilation, model_8_3_groups, model_8_4_weight, model_8_4_bias, model_8_4_running_mean, model_8_4_running_var, model_8_4_eps, model_8_5, model_9_0_weight, model_9_0_bias, model_9_0_stride, model_9_0_padding, model_9_0_dilation, model_9_0_groups, model_9_1_weight, model_9_1_bias, model_9_1_running_mean, model_9_1_running_var, model_9_1_eps, model_9_2, model_9_3_weight, model_9_3_bias, model_9_3_stride, model_9_3_padding, model_9_3_dilation, model_9_3_groups, model_9_4_weight, model_9_4_bias, model_9_4_running_mean, model_9_4_running_var, model_9_4_eps, model_9_5, model_10_0_weight, model_10_0_bias, model_10_0_stride, model_10_0_padding, model_10_0_dilation, model_10_0_groups, model_10_1_weight, model_10_1_bias, model_10_1_running_mean, model_10_1_running_var, model_10_1_eps, model_10_2, model_10_3_weight, model_10_3_bias, model_10_3_stride, model_10_3_padding, model_10_3_dilation, model_10_3_groups, model_10_4_weight, model_10_4_bias, model_10_4_running_mean, model_10_4_running_var, model_10_4_eps, model_10_5, model_11_0_weight, model_11_0_bias, model_11_0_stride, model_11_0_padding, model_11_0_dilation, model_11_0_groups, model_11_1_weight, model_11_1_bias, model_11_1_running_mean, model_11_1_running_var, model_11_1_eps, model_11_2, model_11_3_weight, model_11_3_bias, model_11_3_stride, model_11_3_padding, model_11_3_dilation, model_11_3_groups, model_11_4_weight, model_11_4_bias, model_11_4_running_mean, model_11_4_running_var, model_11_4_eps, model_11_5, model_12_0_weight, model_12_0_bias, model_12_0_stride, model_12_0_padding, model_12_0_dilation, model_12_0_groups, model_12_1_weight, model_12_1_bias, model_12_1_running_mean, model_12_1_running_var, model_12_1_eps, model_12_2, model_12_3_weight, model_12_3_bias, model_12_3_stride, model_12_3_padding, model_12_3_dilation, model_12_3_groups, model_12_4_weight, model_12_4_bias, model_12_4_running_mean, model_12_4_running_var, model_12_4_eps, model_12_5, model_13_0_weight, model_13_0_bias, model_13_0_stride, model_13_0_padding, model_13_0_dilation, model_13_0_groups, model_13_1_weight, model_13_1_bias, model_13_1_running_mean, model_13_1_running_var, model_13_1_eps, model_13_2, model_13_3_weight, model_13_3_bias, model_13_3_stride, model_13_3_padding, model_13_3_dilation, model_13_3_groups, model_13_4_weight, model_13_4_bias, model_13_4_running_mean, model_13_4_running_var, model_13_4_eps, model_13_5, model_14_kernel_size, model_14_stride, model_14_padding, fc_weight, fc_bias
    model_0_0_weight = np.zeros((int(32 * alpha), input_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    model_0_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_0_0_stride = 2
    model_0_0_padding = 1
    model_0_0_dilation = 1
    model_0_0_groups = 1
    model_0_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_0_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_0_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_0_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_0_1_eps = 1e-5
    model_0_2 = None
    model_1_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_1_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_1_0_stride = 1
    model_1_0_padding = 1
    model_1_0_dilation = 1
    model_1_0_groups = int(32 * alpha)
    model_1_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_1_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_1_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_1_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_1_1_eps = 1e-5
    model_1_2 = None
    model_1_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_1_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_1_3_stride = 1
    model_1_3_padding = 0
    model_1_3_dilation = 1
    model_1_3_groups = 1
    model_1_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_1_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_1_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_1_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_1_4_eps = 1e-5
    model_1_5 = None
    model_2_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_2_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_2_0_stride = 1
    model_2_0_padding = 1
    model_2_0_dilation = 1
    model_2_0_groups = int(32 * alpha)
    model_2_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_2_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_2_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_2_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_2_1_eps = 1e-5
    model_2_2 = None
    model_2_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_2_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_2_3_stride = 1
    model_2_3_padding = 0
    model_2_3_dilation = 1
    model_2_3_groups = 1
    model_2_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_2_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_2_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_2_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_2_4_eps = 1e-5
    model_2_5 = None
    model_3_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_3_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_3_0_stride = 1
    model_3_0_padding = 1
    model_3_0_dilation = 1
    model_3_0_groups = int(32 * alpha)
    model_3_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_3_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_3_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_3_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_3_1_eps = 1e-5
    model_3_2 = None
    model_3_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_3_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_3_3_stride = 1
    model_3_3_padding = 0
    model_3_3_dilation = 1
    model_3_3_groups = 1
    model_3_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_3_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_3_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_3_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_3_4_eps = 1e-5
    model_3_5 = None
    model_4_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_4_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_4_0_stride = 1
    model_4_0_padding = 1
    model_4_0_dilation = 1
    model_4_0_groups = int(32 * alpha)
    model_4_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_4_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_4_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_4_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_4_1_eps = 1e-5
    model_4_2 = None
    model_4_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_4_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_4_3_stride = 1
    model_4_3_padding = 0
    model_4_3_dilation = 1
    model_4_3_groups = 1
    model_4_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_4_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_4_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_4_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_4_4_eps = 1e-5
    model_4_5 = None
    model_5_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_5_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_5_0_stride = 1
    model_5_0_padding = 1
    model_5_0_dilation = 1
    model_5_0_groups = int(32 * alpha)
    model_5_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_5_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_5_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_5_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_5_1_eps = 1e-5
    model_5_2 = None
    model_5_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_5_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_5_3_stride = 1
    model_5_3_padding = 0
    model_5_3_dilation = 1
    model_5_3_groups = 1
    model_5_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_5_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_5_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_5_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_5_4_eps = 1e-5
    model_5_5 = None
    model_6_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_6_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_6_0_stride = 1
    model_6_0_padding = 1
    model_6_0_dilation = 1
    model_6_0_groups = int(32 * alpha)
    model_6_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_6_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_6_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_6_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_6_1_eps = 1e-5
    model_6_2 = None
    model_6_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_6_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_6_3_stride = 1
    model_6_3_padding = 0
    model_6_3_dilation = 1
    model_6_3_groups = 1
    model_6_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_6_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_6_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_6_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_6_4_eps = 1e-5
    model_6_5 = None
    model_7_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_7_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_7_0_stride = 1
    model_7_0_padding = 1
    model_7_0_dilation = 1
    model_7_0_groups = int(32 * alpha)
    model_7_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_7_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_7_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_7_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_7_1_eps = 1e-5
    model_7_2 = None
    model_7_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_7_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_7_3_stride = 1
    model_7_3_padding = 0
    model_7_3_dilation = 1
    model_7_3_groups = 1
    model_7_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_7_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_7_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_7_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_7_4_eps = 1e-5
    model_7_5 = None
    model_8_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_8_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_8_0_stride = 1
    model_8_0_padding = 1
    model_8_0_dilation = 1
    model_8_0_groups = int(32 * alpha)
    model_8_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_8_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_8_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_8_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_8_1_eps = 1e-5
    model_8_2 = None
    model_8_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_8_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_8_3_stride = 1
    model_8_3_padding = 0
    model_8_3_dilation = 1
    model_8_3_groups = 1
    model_8_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_8_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_8_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_8_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_8_4_eps = 1e-5
    model_8_5 = None
    model_9_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_9_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_9_0_stride = 1
    model_9_0_padding = 1
    model_9_0_dilation = 1
    model_9_0_groups = int(32 * alpha)
    model_9_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_9_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_9_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_9_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_9_1_eps = 1e-5
    model_9_2 = None
    model_9_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_9_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_9_3_stride = 1
    model_9_3_padding = 0
    model_9_3_dilation = 1
    model_9_3_groups = 1
    model_9_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_9_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_9_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_9_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_9_4_eps = 1e-5
    model_9_5 = None
    model_10_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_10_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_10_0_stride = 1
    model_10_0_padding = 1
    model_10_0_dilation = 1
    model_10_0_groups = int(32 * alpha)
    model_10_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_10_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_10_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_10_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_10_1_eps = 1e-5
    model_10_2 = None
    model_10_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_10_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_10_3_stride = 1
    model_10_3_padding = 0
    model_10_3_dilation = 1
    model_10_3_groups = 1
    model_10_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_10_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_10_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_10_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_10_4_eps = 1e-5
    model_10_5 = None
    model_11_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_11_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_11_0_stride = 1
    model_11_0_padding = 1
    model_11_0_dilation = 1
    model_11_0_groups = int(32 * alpha)
    model_11_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_11_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_11_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_11_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_11_1_eps = 1e-5
    model_11_2 = None
    model_11_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_11_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_11_3_stride = 1
    model_11_3_padding = 0
    model_11_3_dilation = 1
    model_11_3_groups = 1
    model_11_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_11_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_11_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_11_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_11_4_eps = 1e-5
    model_11_5 = None
    model_12_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_12_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_12_0_stride = 1
    model_12_0_padding = 1
    model_12_0_dilation = 1
    model_12_0_groups = int(32 * alpha)
    model_12_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_12_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_12_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_12_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_12_1_eps = 1e-5
    model_12_2 = None
    model_12_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_12_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_12_3_stride = 1
    model_12_3_padding = 0
    model_12_3_dilation = 1
    model_12_3_groups = 1
    model_12_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_12_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_12_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_12_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_12_4_eps = 1e-5
    model_12_5 = None
    model_13_0_weight = np.zeros((int(32 * alpha), int(32 * alpha) // int(32 * alpha)) + _as_tuple(3, 2), dtype=np.float32)
    model_13_0_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_13_0_stride = 1
    model_13_0_padding = 1
    model_13_0_dilation = 1
    model_13_0_groups = int(32 * alpha)
    model_13_1_weight = np.ones((int(32 * alpha),), dtype=np.float32)
    model_13_1_bias = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_13_1_running_mean = np.zeros((int(32 * alpha),), dtype=np.float32)
    model_13_1_running_var = np.ones((int(32 * alpha),), dtype=np.float32)
    model_13_1_eps = 1e-5
    model_13_2 = None
    model_13_3_weight = np.zeros((int(64 * alpha), int(32 * alpha) // 1) + _as_tuple(1, 2), dtype=np.float32)
    model_13_3_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_13_3_stride = 1
    model_13_3_padding = 0
    model_13_3_dilation = 1
    model_13_3_groups = 1
    model_13_4_weight = np.ones((int(64 * alpha),), dtype=np.float32)
    model_13_4_bias = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_13_4_running_mean = np.zeros((int(64 * alpha),), dtype=np.float32)
    model_13_4_running_var = np.ones((int(64 * alpha),), dtype=np.float32)
    model_13_4_eps = 1e-5
    model_13_5 = None
    model_14_kernel_size = 7
    model_14_stride = None
    model_14_padding = 0
    fc_weight = np.zeros((num_classes, int(1024 * alpha)), dtype=np.float32)
    fc_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000, input_channels=3, alpha=1.0):
    x = _avgpool2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(x, model_0_0_weight, model_0_0_bias, model_0_0_stride, model_0_0_padding, model_0_0_dilation, model_0_0_groups), model_0_1_weight, model_0_1_bias, model_0_1_running_mean, model_0_1_running_var, model_0_1_eps), 0), model_1_0_weight, model_1_0_bias, model_1_0_stride, model_1_0_padding, model_1_0_dilation, model_1_0_groups), model_1_1_weight, model_1_1_bias, model_1_1_running_mean, model_1_1_running_var, model_1_1_eps), 0), model_1_3_weight, model_1_3_bias, model_1_3_stride, model_1_3_padding, model_1_3_dilation, model_1_3_groups), model_1_4_weight, model_1_4_bias, model_1_4_running_mean, model_1_4_running_var, model_1_4_eps), 0), model_2_0_weight, model_2_0_bias, model_2_0_stride, model_2_0_padding, model_2_0_dilation, model_2_0_groups), model_2_1_weight, model_2_1_bias, model_2_1_running_mean, model_2_1_running_var, model_2_1_eps), 0), model_2_3_weight, model_2_3_bias, model_2_3_stride, model_2_3_padding, model_2_3_dilation, model_2_3_groups), model_2_4_weight, model_2_4_bias, model_2_4_running_mean, model_2_4_running_var, model_2_4_eps), 0), model_3_0_weight, model_3_0_bias, model_3_0_stride, model_3_0_padding, model_3_0_dilation, model_3_0_groups), model_3_1_weight, model_3_1_bias, model_3_1_running_mean, model_3_1_running_var, model_3_1_eps), 0), model_3_3_weight, model_3_3_bias, model_3_3_stride, model_3_3_padding, model_3_3_dilation, model_3_3_groups), model_3_4_weight, model_3_4_bias, model_3_4_running_mean, model_3_4_running_var, model_3_4_eps), 0), model_4_0_weight, model_4_0_bias, model_4_0_stride, model_4_0_padding, model_4_0_dilation, model_4_0_groups), model_4_1_weight, model_4_1_bias, model_4_1_running_mean, model_4_1_running_var, model_4_1_eps), 0), model_4_3_weight, model_4_3_bias, model_4_3_stride, model_4_3_padding, model_4_3_dilation, model_4_3_groups), model_4_4_weight, model_4_4_bias, model_4_4_running_mean, model_4_4_running_var, model_4_4_eps), 0), model_5_0_weight, model_5_0_bias, model_5_0_stride, model_5_0_padding, model_5_0_dilation, model_5_0_groups), model_5_1_weight, model_5_1_bias, model_5_1_running_mean, model_5_1_running_var, model_5_1_eps), 0), model_5_3_weight, model_5_3_bias, model_5_3_stride, model_5_3_padding, model_5_3_dilation, model_5_3_groups), model_5_4_weight, model_5_4_bias, model_5_4_running_mean, model_5_4_running_var, model_5_4_eps), 0), model_6_0_weight, model_6_0_bias, model_6_0_stride, model_6_0_padding, model_6_0_dilation, model_6_0_groups), model_6_1_weight, model_6_1_bias, model_6_1_running_mean, model_6_1_running_var, model_6_1_eps), 0), model_6_3_weight, model_6_3_bias, model_6_3_stride, model_6_3_padding, model_6_3_dilation, model_6_3_groups), model_6_4_weight, model_6_4_bias, model_6_4_running_mean, model_6_4_running_var, model_6_4_eps), 0), model_7_0_weight, model_7_0_bias, model_7_0_stride, model_7_0_padding, model_7_0_dilation, model_7_0_groups), model_7_1_weight, model_7_1_bias, model_7_1_running_mean, model_7_1_running_var, model_7_1_eps), 0), model_7_3_weight, model_7_3_bias, model_7_3_stride, model_7_3_padding, model_7_3_dilation, model_7_3_groups), model_7_4_weight, model_7_4_bias, model_7_4_running_mean, model_7_4_running_var, model_7_4_eps), 0), model_8_0_weight, model_8_0_bias, model_8_0_stride, model_8_0_padding, model_8_0_dilation, model_8_0_groups), model_8_1_weight, model_8_1_bias, model_8_1_running_mean, model_8_1_running_var, model_8_1_eps), 0), model_8_3_weight, model_8_3_bias, model_8_3_stride, model_8_3_padding, model_8_3_dilation, model_8_3_groups), model_8_4_weight, model_8_4_bias, model_8_4_running_mean, model_8_4_running_var, model_8_4_eps), 0), model_9_0_weight, model_9_0_bias, model_9_0_stride, model_9_0_padding, model_9_0_dilation, model_9_0_groups), model_9_1_weight, model_9_1_bias, model_9_1_running_mean, model_9_1_running_var, model_9_1_eps), 0), model_9_3_weight, model_9_3_bias, model_9_3_stride, model_9_3_padding, model_9_3_dilation, model_9_3_groups), model_9_4_weight, model_9_4_bias, model_9_4_running_mean, model_9_4_running_var, model_9_4_eps), 0), model_10_0_weight, model_10_0_bias, model_10_0_stride, model_10_0_padding, model_10_0_dilation, model_10_0_groups), model_10_1_weight, model_10_1_bias, model_10_1_running_mean, model_10_1_running_var, model_10_1_eps), 0), model_10_3_weight, model_10_3_bias, model_10_3_stride, model_10_3_padding, model_10_3_dilation, model_10_3_groups), model_10_4_weight, model_10_4_bias, model_10_4_running_mean, model_10_4_running_var, model_10_4_eps), 0), model_11_0_weight, model_11_0_bias, model_11_0_stride, model_11_0_padding, model_11_0_dilation, model_11_0_groups), model_11_1_weight, model_11_1_bias, model_11_1_running_mean, model_11_1_running_var, model_11_1_eps), 0), model_11_3_weight, model_11_3_bias, model_11_3_stride, model_11_3_padding, model_11_3_dilation, model_11_3_groups), model_11_4_weight, model_11_4_bias, model_11_4_running_mean, model_11_4_running_var, model_11_4_eps), 0), model_12_0_weight, model_12_0_bias, model_12_0_stride, model_12_0_padding, model_12_0_dilation, model_12_0_groups), model_12_1_weight, model_12_1_bias, model_12_1_running_mean, model_12_1_running_var, model_12_1_eps), 0), model_12_3_weight, model_12_3_bias, model_12_3_stride, model_12_3_padding, model_12_3_dilation, model_12_3_groups), model_12_4_weight, model_12_4_bias, model_12_4_running_mean, model_12_4_running_var, model_12_4_eps), 0), model_13_0_weight, model_13_0_bias, model_13_0_stride, model_13_0_padding, model_13_0_dilation, model_13_0_groups), model_13_1_weight, model_13_1_bias, model_13_1_running_mean, model_13_1_running_var, model_13_1_eps), 0), model_13_3_weight, model_13_3_bias, model_13_3_stride, model_13_3_padding, model_13_3_dilation, model_13_3_groups), model_13_4_weight, model_13_4_bias, model_13_4_running_mean, model_13_4_running_var, model_13_4_eps), 0), model_14_kernel_size, model_14_stride, model_14_padding)
    x = np.reshape(x, (x.shape[0], (-1)))
    x = ((x) @ fc_weight.T + fc_bias)
    return x

