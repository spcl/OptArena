import numpy as np

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

def init(num_classes):
    global conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups, fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias
    conv1_weight = np.zeros((6, 1 // 1) + _as_tuple(5, 2), dtype=np.float32)
    conv1_bias = np.zeros((6,), dtype=np.float32)
    conv1_stride = 1
    conv1_padding = 0
    conv1_dilation = 1
    conv1_groups = 1
    conv2_weight = np.zeros((16, 6 // 1) + _as_tuple(5, 2), dtype=np.float32)
    conv2_bias = np.zeros((16,), dtype=np.float32)
    conv2_stride = 1
    conv2_padding = 0
    conv2_dilation = 1
    conv2_groups = 1
    fc1_weight = np.zeros((120, 16 * 5 * 5), dtype=np.float32)
    fc1_bias = np.zeros((120,), dtype=np.float32) if True else np.zeros((120,), dtype=np.float32)
    fc2_weight = np.zeros((84, 120), dtype=np.float32)
    fc2_bias = np.zeros((84,), dtype=np.float32) if True else np.zeros((84,), dtype=np.float32)
    fc3_weight = np.zeros((num_classes, 84), dtype=np.float32)
    fc3_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes):
    x = np.maximum(_conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups), 0)
    x = _maxpool2d(x, 2, 2, 0)
    x = np.maximum(_conv2d(x, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups), 0)
    x = _maxpool2d(x, 2, 2, 0)
    x = np.reshape(x, ((-1), ((16 * 5) * 5)))
    x = np.maximum(((x) @ fc1_weight.T + fc1_bias), 0)
    x = np.maximum(((x) @ fc2_weight.T + fc2_bias), 0)
    x = ((x) @ fc3_weight.T + fc3_bias)
    return x

