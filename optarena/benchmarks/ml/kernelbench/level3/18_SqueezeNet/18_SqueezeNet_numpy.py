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

def _maxpool2d(x, kernel_size, stride, padding):
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size,)
    if stride is None: stride = kernel_size
    if isinstance(stride, int): stride = (stride, stride,)
    if isinstance(padding, int): padding = (padding, padding,)
    padded_shape = (x.shape[0], x.shape[1]) + tuple(x.shape[i + 2] + 2 * padding[i] for i in range(2))
    fill = -np.inf if "max" == "max" else 0.0
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
                    out[b, c, oy, ox] = np.max(window)
    return out

def _features_3_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_3_expand1x1_weight, features_3_expand1x1_bias, features_3_expand1x1_stride, features_3_expand1x1_padding, features_3_expand1x1_dilation, features_3_expand1x1_groups), 0), np.maximum(_conv2d(x, features_3_expand3x3_weight, features_3_expand3x3_bias, features_3_expand3x3_stride, features_3_expand3x3_padding, features_3_expand3x3_dilation, features_3_expand3x3_groups), 0)), axis=1)

def _features_4_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_4_expand1x1_weight, features_4_expand1x1_bias, features_4_expand1x1_stride, features_4_expand1x1_padding, features_4_expand1x1_dilation, features_4_expand1x1_groups), 0), np.maximum(_conv2d(x, features_4_expand3x3_weight, features_4_expand3x3_bias, features_4_expand3x3_stride, features_4_expand3x3_padding, features_4_expand3x3_dilation, features_4_expand3x3_groups), 0)), axis=1)

def _features_5_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_5_expand1x1_weight, features_5_expand1x1_bias, features_5_expand1x1_stride, features_5_expand1x1_padding, features_5_expand1x1_dilation, features_5_expand1x1_groups), 0), np.maximum(_conv2d(x, features_5_expand3x3_weight, features_5_expand3x3_bias, features_5_expand3x3_stride, features_5_expand3x3_padding, features_5_expand3x3_dilation, features_5_expand3x3_groups), 0)), axis=1)

def _features_7_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_7_expand1x1_weight, features_7_expand1x1_bias, features_7_expand1x1_stride, features_7_expand1x1_padding, features_7_expand1x1_dilation, features_7_expand1x1_groups), 0), np.maximum(_conv2d(x, features_7_expand3x3_weight, features_7_expand3x3_bias, features_7_expand3x3_stride, features_7_expand3x3_padding, features_7_expand3x3_dilation, features_7_expand3x3_groups), 0)), axis=1)

def _features_8_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_8_expand1x1_weight, features_8_expand1x1_bias, features_8_expand1x1_stride, features_8_expand1x1_padding, features_8_expand1x1_dilation, features_8_expand1x1_groups), 0), np.maximum(_conv2d(x, features_8_expand3x3_weight, features_8_expand3x3_bias, features_8_expand3x3_stride, features_8_expand3x3_padding, features_8_expand3x3_dilation, features_8_expand3x3_groups), 0)), axis=1)

def _features_9_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_9_expand1x1_weight, features_9_expand1x1_bias, features_9_expand1x1_stride, features_9_expand1x1_padding, features_9_expand1x1_dilation, features_9_expand1x1_groups), 0), np.maximum(_conv2d(x, features_9_expand3x3_weight, features_9_expand3x3_bias, features_9_expand3x3_stride, features_9_expand3x3_padding, features_9_expand3x3_dilation, features_9_expand3x3_groups), 0)), axis=1)

def _features_10_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_10_expand1x1_weight, features_10_expand1x1_bias, features_10_expand1x1_stride, features_10_expand1x1_padding, features_10_expand1x1_dilation, features_10_expand1x1_groups), 0), np.maximum(_conv2d(x, features_10_expand3x3_weight, features_10_expand3x3_bias, features_10_expand3x3_stride, features_10_expand3x3_padding, features_10_expand3x3_dilation, features_10_expand3x3_groups), 0)), axis=1)

def _features_12_forward(x):
    x = np.maximum(np.squeeze(self, axis=x), 0)
    return np.concatenate((np.maximum(_conv2d(x, features_12_expand1x1_weight, features_12_expand1x1_bias, features_12_expand1x1_stride, features_12_expand1x1_padding, features_12_expand1x1_dilation, features_12_expand1x1_groups), 0), np.maximum(_conv2d(x, features_12_expand3x3_weight, features_12_expand3x3_bias, features_12_expand3x3_stride, features_12_expand3x3_padding, features_12_expand3x3_dilation, features_12_expand3x3_groups), 0)), axis=1)

def init(num_classes=1000):
    global features_0_weight, features_0_bias, features_0_stride, features_0_padding, features_0_dilation, features_0_groups, features_1, features_2_kernel_size, features_2_stride, features_2_padding, features_3_squeeze_weight, features_3_squeeze_bias, features_3_squeeze_stride, features_3_squeeze_padding, features_3_squeeze_dilation, features_3_squeeze_groups, features_3_squeeze_activation, features_3_expand1x1_weight, features_3_expand1x1_bias, features_3_expand1x1_stride, features_3_expand1x1_padding, features_3_expand1x1_dilation, features_3_expand1x1_groups, features_3_expand1x1_activation, features_3_expand3x3_weight, features_3_expand3x3_bias, features_3_expand3x3_stride, features_3_expand3x3_padding, features_3_expand3x3_dilation, features_3_expand3x3_groups, features_3_expand3x3_activation, features_4_squeeze_weight, features_4_squeeze_bias, features_4_squeeze_stride, features_4_squeeze_padding, features_4_squeeze_dilation, features_4_squeeze_groups, features_4_squeeze_activation, features_4_expand1x1_weight, features_4_expand1x1_bias, features_4_expand1x1_stride, features_4_expand1x1_padding, features_4_expand1x1_dilation, features_4_expand1x1_groups, features_4_expand1x1_activation, features_4_expand3x3_weight, features_4_expand3x3_bias, features_4_expand3x3_stride, features_4_expand3x3_padding, features_4_expand3x3_dilation, features_4_expand3x3_groups, features_4_expand3x3_activation, features_5_squeeze_weight, features_5_squeeze_bias, features_5_squeeze_stride, features_5_squeeze_padding, features_5_squeeze_dilation, features_5_squeeze_groups, features_5_squeeze_activation, features_5_expand1x1_weight, features_5_expand1x1_bias, features_5_expand1x1_stride, features_5_expand1x1_padding, features_5_expand1x1_dilation, features_5_expand1x1_groups, features_5_expand1x1_activation, features_5_expand3x3_weight, features_5_expand3x3_bias, features_5_expand3x3_stride, features_5_expand3x3_padding, features_5_expand3x3_dilation, features_5_expand3x3_groups, features_5_expand3x3_activation, features_6_kernel_size, features_6_stride, features_6_padding, features_7_squeeze_weight, features_7_squeeze_bias, features_7_squeeze_stride, features_7_squeeze_padding, features_7_squeeze_dilation, features_7_squeeze_groups, features_7_squeeze_activation, features_7_expand1x1_weight, features_7_expand1x1_bias, features_7_expand1x1_stride, features_7_expand1x1_padding, features_7_expand1x1_dilation, features_7_expand1x1_groups, features_7_expand1x1_activation, features_7_expand3x3_weight, features_7_expand3x3_bias, features_7_expand3x3_stride, features_7_expand3x3_padding, features_7_expand3x3_dilation, features_7_expand3x3_groups, features_7_expand3x3_activation, features_8_squeeze_weight, features_8_squeeze_bias, features_8_squeeze_stride, features_8_squeeze_padding, features_8_squeeze_dilation, features_8_squeeze_groups, features_8_squeeze_activation, features_8_expand1x1_weight, features_8_expand1x1_bias, features_8_expand1x1_stride, features_8_expand1x1_padding, features_8_expand1x1_dilation, features_8_expand1x1_groups, features_8_expand1x1_activation, features_8_expand3x3_weight, features_8_expand3x3_bias, features_8_expand3x3_stride, features_8_expand3x3_padding, features_8_expand3x3_dilation, features_8_expand3x3_groups, features_8_expand3x3_activation, features_9_squeeze_weight, features_9_squeeze_bias, features_9_squeeze_stride, features_9_squeeze_padding, features_9_squeeze_dilation, features_9_squeeze_groups, features_9_squeeze_activation, features_9_expand1x1_weight, features_9_expand1x1_bias, features_9_expand1x1_stride, features_9_expand1x1_padding, features_9_expand1x1_dilation, features_9_expand1x1_groups, features_9_expand1x1_activation, features_9_expand3x3_weight, features_9_expand3x3_bias, features_9_expand3x3_stride, features_9_expand3x3_padding, features_9_expand3x3_dilation, features_9_expand3x3_groups, features_9_expand3x3_activation, features_10_squeeze_weight, features_10_squeeze_bias, features_10_squeeze_stride, features_10_squeeze_padding, features_10_squeeze_dilation, features_10_squeeze_groups, features_10_squeeze_activation, features_10_expand1x1_weight, features_10_expand1x1_bias, features_10_expand1x1_stride, features_10_expand1x1_padding, features_10_expand1x1_dilation, features_10_expand1x1_groups, features_10_expand1x1_activation, features_10_expand3x3_weight, features_10_expand3x3_bias, features_10_expand3x3_stride, features_10_expand3x3_padding, features_10_expand3x3_dilation, features_10_expand3x3_groups, features_10_expand3x3_activation, features_11_kernel_size, features_11_stride, features_11_padding, features_12_squeeze_weight, features_12_squeeze_bias, features_12_squeeze_stride, features_12_squeeze_padding, features_12_squeeze_dilation, features_12_squeeze_groups, features_12_squeeze_activation, features_12_expand1x1_weight, features_12_expand1x1_bias, features_12_expand1x1_stride, features_12_expand1x1_padding, features_12_expand1x1_dilation, features_12_expand1x1_groups, features_12_expand1x1_activation, features_12_expand3x3_weight, features_12_expand3x3_bias, features_12_expand3x3_stride, features_12_expand3x3_padding, features_12_expand3x3_dilation, features_12_expand3x3_groups, features_12_expand3x3_activation, classifier_0, classifier_1_weight, classifier_1_bias, classifier_1_stride, classifier_1_padding, classifier_1_dilation, classifier_1_groups, classifier_2, classifier_3_output_size
    features_0_weight = np.zeros((96, 3 // 1) + _as_tuple(7, 2), dtype=np.float32)
    features_0_bias = np.zeros((96,), dtype=np.float32)
    features_0_stride = 2
    features_0_padding = 0
    features_0_dilation = 1
    features_0_groups = 1
    features_1 = None
    features_2_kernel_size = 3
    features_2_stride = 2
    features_2_padding = 0
    features_3_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_3_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_3_squeeze_stride = 1
    features_3_squeeze_padding = 0
    features_3_squeeze_dilation = 1
    features_3_squeeze_groups = 1
    features_3_squeeze_activation = None
    features_3_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_3_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_3_expand1x1_stride = 1
    features_3_expand1x1_padding = 0
    features_3_expand1x1_dilation = 1
    features_3_expand1x1_groups = 1
    features_3_expand1x1_activation = None
    features_3_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_3_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_3_expand3x3_stride = 1
    features_3_expand3x3_padding = 1
    features_3_expand3x3_dilation = 1
    features_3_expand3x3_groups = 1
    features_3_expand3x3_activation = None
    features_4_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_4_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_4_squeeze_stride = 1
    features_4_squeeze_padding = 0
    features_4_squeeze_dilation = 1
    features_4_squeeze_groups = 1
    features_4_squeeze_activation = None
    features_4_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_4_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_4_expand1x1_stride = 1
    features_4_expand1x1_padding = 0
    features_4_expand1x1_dilation = 1
    features_4_expand1x1_groups = 1
    features_4_expand1x1_activation = None
    features_4_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_4_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_4_expand3x3_stride = 1
    features_4_expand3x3_padding = 1
    features_4_expand3x3_dilation = 1
    features_4_expand3x3_groups = 1
    features_4_expand3x3_activation = None
    features_5_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_5_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_5_squeeze_stride = 1
    features_5_squeeze_padding = 0
    features_5_squeeze_dilation = 1
    features_5_squeeze_groups = 1
    features_5_squeeze_activation = None
    features_5_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_5_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_5_expand1x1_stride = 1
    features_5_expand1x1_padding = 0
    features_5_expand1x1_dilation = 1
    features_5_expand1x1_groups = 1
    features_5_expand1x1_activation = None
    features_5_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_5_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_5_expand3x3_stride = 1
    features_5_expand3x3_padding = 1
    features_5_expand3x3_dilation = 1
    features_5_expand3x3_groups = 1
    features_5_expand3x3_activation = None
    features_6_kernel_size = 3
    features_6_stride = 2
    features_6_padding = 0
    features_7_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_7_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_7_squeeze_stride = 1
    features_7_squeeze_padding = 0
    features_7_squeeze_dilation = 1
    features_7_squeeze_groups = 1
    features_7_squeeze_activation = None
    features_7_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_7_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_7_expand1x1_stride = 1
    features_7_expand1x1_padding = 0
    features_7_expand1x1_dilation = 1
    features_7_expand1x1_groups = 1
    features_7_expand1x1_activation = None
    features_7_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_7_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_7_expand3x3_stride = 1
    features_7_expand3x3_padding = 1
    features_7_expand3x3_dilation = 1
    features_7_expand3x3_groups = 1
    features_7_expand3x3_activation = None
    features_8_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_8_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_8_squeeze_stride = 1
    features_8_squeeze_padding = 0
    features_8_squeeze_dilation = 1
    features_8_squeeze_groups = 1
    features_8_squeeze_activation = None
    features_8_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_8_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_8_expand1x1_stride = 1
    features_8_expand1x1_padding = 0
    features_8_expand1x1_dilation = 1
    features_8_expand1x1_groups = 1
    features_8_expand1x1_activation = None
    features_8_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_8_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_8_expand3x3_stride = 1
    features_8_expand3x3_padding = 1
    features_8_expand3x3_dilation = 1
    features_8_expand3x3_groups = 1
    features_8_expand3x3_activation = None
    features_9_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_9_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_9_squeeze_stride = 1
    features_9_squeeze_padding = 0
    features_9_squeeze_dilation = 1
    features_9_squeeze_groups = 1
    features_9_squeeze_activation = None
    features_9_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_9_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_9_expand1x1_stride = 1
    features_9_expand1x1_padding = 0
    features_9_expand1x1_dilation = 1
    features_9_expand1x1_groups = 1
    features_9_expand1x1_activation = None
    features_9_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_9_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_9_expand3x3_stride = 1
    features_9_expand3x3_padding = 1
    features_9_expand3x3_dilation = 1
    features_9_expand3x3_groups = 1
    features_9_expand3x3_activation = None
    features_10_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_10_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_10_squeeze_stride = 1
    features_10_squeeze_padding = 0
    features_10_squeeze_dilation = 1
    features_10_squeeze_groups = 1
    features_10_squeeze_activation = None
    features_10_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_10_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_10_expand1x1_stride = 1
    features_10_expand1x1_padding = 0
    features_10_expand1x1_dilation = 1
    features_10_expand1x1_groups = 1
    features_10_expand1x1_activation = None
    features_10_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_10_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_10_expand3x3_stride = 1
    features_10_expand3x3_padding = 1
    features_10_expand3x3_dilation = 1
    features_10_expand3x3_groups = 1
    features_10_expand3x3_activation = None
    features_11_kernel_size = 3
    features_11_stride = 2
    features_11_padding = 0
    features_12_squeeze_weight = np.zeros((16, 96 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_12_squeeze_bias = np.zeros((16,), dtype=np.float32)
    features_12_squeeze_stride = 1
    features_12_squeeze_padding = 0
    features_12_squeeze_dilation = 1
    features_12_squeeze_groups = 1
    features_12_squeeze_activation = None
    features_12_expand1x1_weight = np.zeros((64, 16 // 1) + _as_tuple(1, 2), dtype=np.float32)
    features_12_expand1x1_bias = np.zeros((64,), dtype=np.float32)
    features_12_expand1x1_stride = 1
    features_12_expand1x1_padding = 0
    features_12_expand1x1_dilation = 1
    features_12_expand1x1_groups = 1
    features_12_expand1x1_activation = None
    features_12_expand3x3_weight = np.zeros((64, 16 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_12_expand3x3_bias = np.zeros((64,), dtype=np.float32)
    features_12_expand3x3_stride = 1
    features_12_expand3x3_padding = 1
    features_12_expand3x3_dilation = 1
    features_12_expand3x3_groups = 1
    features_12_expand3x3_activation = None
    classifier_0 = None
    classifier_1_weight = np.zeros((num_classes, 512 // 1) + _as_tuple(1, 2), dtype=np.float32)
    classifier_1_bias = np.zeros((num_classes,), dtype=np.float32)
    classifier_1_stride = 1
    classifier_1_padding = 0
    classifier_1_dilation = 1
    classifier_1_groups = 1
    classifier_2 = None
    classifier_3_output_size = (1, 1)

def forward(x, num_classes=1000):
    x = _features_12_forward(_maxpool2d(_features_10_forward(_features_9_forward(_features_8_forward(_features_7_forward(_maxpool2d(_features_5_forward(_features_4_forward(_features_3_forward(_maxpool2d(np.maximum(_conv2d(x, features_0_weight, features_0_bias, features_0_stride, features_0_padding, features_0_dilation, features_0_groups), 0), features_2_kernel_size, features_2_stride, features_2_padding)))), features_6_kernel_size, features_6_stride, features_6_padding))))), features_11_kernel_size, features_11_stride, features_11_padding))
    x = _adaptive_avg_pool2d(np.maximum(_conv2d(x, classifier_1_weight, classifier_1_bias, classifier_1_stride, classifier_1_padding, classifier_1_dilation, classifier_1_groups), 0), classifier_3_output_size)
    return np.reshape(x, (x.shape[0], -1))

