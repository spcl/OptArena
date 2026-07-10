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

def init(input_channels, stages, block_widths, output_classes):
    global feature_extractor_0_0_weight, feature_extractor_0_0_bias, feature_extractor_0_0_stride, feature_extractor_0_0_padding, feature_extractor_0_0_dilation, feature_extractor_0_0_groups, feature_extractor_0_1_weight, feature_extractor_0_1_bias, feature_extractor_0_1_running_mean, feature_extractor_0_1_running_var, feature_extractor_0_1_eps, feature_extractor_0_2, feature_extractor_0_3_weight, feature_extractor_0_3_bias, feature_extractor_0_3_stride, feature_extractor_0_3_padding, feature_extractor_0_3_dilation, feature_extractor_0_3_groups, feature_extractor_0_4_weight, feature_extractor_0_4_bias, feature_extractor_0_4_running_mean, feature_extractor_0_4_running_var, feature_extractor_0_4_eps, feature_extractor_0_5, feature_extractor_0_6_kernel_size, feature_extractor_0_6_stride, feature_extractor_0_6_padding, feature_extractor_1_0_weight, feature_extractor_1_0_bias, feature_extractor_1_0_stride, feature_extractor_1_0_padding, feature_extractor_1_0_dilation, feature_extractor_1_0_groups, feature_extractor_1_1_weight, feature_extractor_1_1_bias, feature_extractor_1_1_running_mean, feature_extractor_1_1_running_var, feature_extractor_1_1_eps, feature_extractor_1_2, feature_extractor_1_3_weight, feature_extractor_1_3_bias, feature_extractor_1_3_stride, feature_extractor_1_3_padding, feature_extractor_1_3_dilation, feature_extractor_1_3_groups, feature_extractor_1_4_weight, feature_extractor_1_4_bias, feature_extractor_1_4_running_mean, feature_extractor_1_4_running_var, feature_extractor_1_4_eps, feature_extractor_1_5, feature_extractor_1_6_kernel_size, feature_extractor_1_6_stride, feature_extractor_1_6_padding, feature_extractor_2_0_weight, feature_extractor_2_0_bias, feature_extractor_2_0_stride, feature_extractor_2_0_padding, feature_extractor_2_0_dilation, feature_extractor_2_0_groups, feature_extractor_2_1_weight, feature_extractor_2_1_bias, feature_extractor_2_1_running_mean, feature_extractor_2_1_running_var, feature_extractor_2_1_eps, feature_extractor_2_2, feature_extractor_2_3_weight, feature_extractor_2_3_bias, feature_extractor_2_3_stride, feature_extractor_2_3_padding, feature_extractor_2_3_dilation, feature_extractor_2_3_groups, feature_extractor_2_4_weight, feature_extractor_2_4_bias, feature_extractor_2_4_running_mean, feature_extractor_2_4_running_var, feature_extractor_2_4_eps, feature_extractor_2_5, feature_extractor_2_6_kernel_size, feature_extractor_2_6_stride, feature_extractor_2_6_padding, fc_weight, fc_bias
    feature_extractor_0_0_weight = np.zeros((block_widths[0], input_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    feature_extractor_0_0_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_0_0_stride = 1
    feature_extractor_0_0_padding = 1
    feature_extractor_0_0_dilation = 1
    feature_extractor_0_0_groups = 1
    feature_extractor_0_1_weight = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_0_1_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_0_1_running_mean = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_0_1_running_var = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_0_1_eps = 1e-5
    feature_extractor_0_2 = None
    feature_extractor_0_3_weight = np.zeros((block_widths[0], block_widths[0] // 1) + _as_tuple(3, 2), dtype=np.float32)
    feature_extractor_0_3_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_0_3_stride = 1
    feature_extractor_0_3_padding = 1
    feature_extractor_0_3_dilation = 1
    feature_extractor_0_3_groups = 1
    feature_extractor_0_4_weight = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_0_4_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_0_4_running_mean = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_0_4_running_var = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_0_4_eps = 1e-5
    feature_extractor_0_5 = None
    feature_extractor_0_6_kernel_size = 2
    feature_extractor_0_6_stride = 2
    feature_extractor_0_6_padding = 0
    feature_extractor_1_0_weight = np.zeros((block_widths[0], input_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    feature_extractor_1_0_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_1_0_stride = 1
    feature_extractor_1_0_padding = 1
    feature_extractor_1_0_dilation = 1
    feature_extractor_1_0_groups = 1
    feature_extractor_1_1_weight = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_1_1_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_1_1_running_mean = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_1_1_running_var = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_1_1_eps = 1e-5
    feature_extractor_1_2 = None
    feature_extractor_1_3_weight = np.zeros((block_widths[0], block_widths[0] // 1) + _as_tuple(3, 2), dtype=np.float32)
    feature_extractor_1_3_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_1_3_stride = 1
    feature_extractor_1_3_padding = 1
    feature_extractor_1_3_dilation = 1
    feature_extractor_1_3_groups = 1
    feature_extractor_1_4_weight = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_1_4_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_1_4_running_mean = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_1_4_running_var = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_1_4_eps = 1e-5
    feature_extractor_1_5 = None
    feature_extractor_1_6_kernel_size = 2
    feature_extractor_1_6_stride = 2
    feature_extractor_1_6_padding = 0
    feature_extractor_2_0_weight = np.zeros((block_widths[0], input_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    feature_extractor_2_0_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_2_0_stride = 1
    feature_extractor_2_0_padding = 1
    feature_extractor_2_0_dilation = 1
    feature_extractor_2_0_groups = 1
    feature_extractor_2_1_weight = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_2_1_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_2_1_running_mean = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_2_1_running_var = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_2_1_eps = 1e-5
    feature_extractor_2_2 = None
    feature_extractor_2_3_weight = np.zeros((block_widths[0], block_widths[0] // 1) + _as_tuple(3, 2), dtype=np.float32)
    feature_extractor_2_3_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_2_3_stride = 1
    feature_extractor_2_3_padding = 1
    feature_extractor_2_3_dilation = 1
    feature_extractor_2_3_groups = 1
    feature_extractor_2_4_weight = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_2_4_bias = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_2_4_running_mean = np.zeros((block_widths[0],), dtype=np.float32)
    feature_extractor_2_4_running_var = np.ones((block_widths[0],), dtype=np.float32)
    feature_extractor_2_4_eps = 1e-5
    feature_extractor_2_5 = None
    feature_extractor_2_6_kernel_size = 2
    feature_extractor_2_6_stride = 2
    feature_extractor_2_6_padding = 0
    fc_weight = np.zeros((output_classes, block_widths[-1]), dtype=np.float32)
    fc_bias = np.zeros((output_classes,), dtype=np.float32) if True else np.zeros((output_classes,), dtype=np.float32)

def forward(x, input_channels, stages, block_widths, output_classes):
    x = _maxpool2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(_maxpool2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(_maxpool2d(np.maximum(_batch_norm(_conv2d(np.maximum(_batch_norm(_conv2d(x, feature_extractor_0_0_weight, feature_extractor_0_0_bias, feature_extractor_0_0_stride, feature_extractor_0_0_padding, feature_extractor_0_0_dilation, feature_extractor_0_0_groups), feature_extractor_0_1_weight, feature_extractor_0_1_bias, feature_extractor_0_1_running_mean, feature_extractor_0_1_running_var, feature_extractor_0_1_eps), 0), feature_extractor_0_3_weight, feature_extractor_0_3_bias, feature_extractor_0_3_stride, feature_extractor_0_3_padding, feature_extractor_0_3_dilation, feature_extractor_0_3_groups), feature_extractor_0_4_weight, feature_extractor_0_4_bias, feature_extractor_0_4_running_mean, feature_extractor_0_4_running_var, feature_extractor_0_4_eps), 0), feature_extractor_0_6_kernel_size, feature_extractor_0_6_stride, feature_extractor_0_6_padding), feature_extractor_1_0_weight, feature_extractor_1_0_bias, feature_extractor_1_0_stride, feature_extractor_1_0_padding, feature_extractor_1_0_dilation, feature_extractor_1_0_groups), feature_extractor_1_1_weight, feature_extractor_1_1_bias, feature_extractor_1_1_running_mean, feature_extractor_1_1_running_var, feature_extractor_1_1_eps), 0), feature_extractor_1_3_weight, feature_extractor_1_3_bias, feature_extractor_1_3_stride, feature_extractor_1_3_padding, feature_extractor_1_3_dilation, feature_extractor_1_3_groups), feature_extractor_1_4_weight, feature_extractor_1_4_bias, feature_extractor_1_4_running_mean, feature_extractor_1_4_running_var, feature_extractor_1_4_eps), 0), feature_extractor_1_6_kernel_size, feature_extractor_1_6_stride, feature_extractor_1_6_padding), feature_extractor_2_0_weight, feature_extractor_2_0_bias, feature_extractor_2_0_stride, feature_extractor_2_0_padding, feature_extractor_2_0_dilation, feature_extractor_2_0_groups), feature_extractor_2_1_weight, feature_extractor_2_1_bias, feature_extractor_2_1_running_mean, feature_extractor_2_1_running_var, feature_extractor_2_1_eps), 0), feature_extractor_2_3_weight, feature_extractor_2_3_bias, feature_extractor_2_3_stride, feature_extractor_2_3_padding, feature_extractor_2_3_dilation, feature_extractor_2_3_groups), feature_extractor_2_4_weight, feature_extractor_2_4_bias, feature_extractor_2_4_running_mean, feature_extractor_2_4_running_var, feature_extractor_2_4_eps), 0), feature_extractor_2_6_kernel_size, feature_extractor_2_6_stride, feature_extractor_2_6_padding)
    x = np.mean(x, axis=(2, 3), keepdims=False)
    x = ((x) @ fc_weight.T + fc_bias)
    return x

