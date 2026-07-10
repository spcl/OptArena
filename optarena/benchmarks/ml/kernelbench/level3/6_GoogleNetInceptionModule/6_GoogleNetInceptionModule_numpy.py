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

def init(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
    global branch1x1_weight, branch1x1_bias, branch1x1_stride, branch1x1_padding, branch1x1_dilation, branch1x1_groups, branch3x3_0_weight, branch3x3_0_bias, branch3x3_0_stride, branch3x3_0_padding, branch3x3_0_dilation, branch3x3_0_groups, branch3x3_1_weight, branch3x3_1_bias, branch3x3_1_stride, branch3x3_1_padding, branch3x3_1_dilation, branch3x3_1_groups, branch5x5_0_weight, branch5x5_0_bias, branch5x5_0_stride, branch5x5_0_padding, branch5x5_0_dilation, branch5x5_0_groups, branch5x5_1_weight, branch5x5_1_bias, branch5x5_1_stride, branch5x5_1_padding, branch5x5_1_dilation, branch5x5_1_groups, branch_pool_0_kernel_size, branch_pool_0_stride, branch_pool_0_padding, branch_pool_1_weight, branch_pool_1_bias, branch_pool_1_stride, branch_pool_1_padding, branch_pool_1_dilation, branch_pool_1_groups
    branch1x1_weight = np.zeros((out_1x1, in_channels // 1) + _as_tuple(1, 2), dtype=np.float32)
    branch1x1_bias = np.zeros((out_1x1,), dtype=np.float32)
    branch1x1_stride = 1
    branch1x1_padding = 0
    branch1x1_dilation = 1
    branch1x1_groups = 1
    branch3x3_0_weight = np.zeros((reduce_3x3, in_channels // 1) + _as_tuple(1, 2), dtype=np.float32)
    branch3x3_0_bias = np.zeros((reduce_3x3,), dtype=np.float32)
    branch3x3_0_stride = 1
    branch3x3_0_padding = 0
    branch3x3_0_dilation = 1
    branch3x3_0_groups = 1
    branch3x3_1_weight = np.zeros((out_3x3, reduce_3x3 // 1) + _as_tuple(3, 2), dtype=np.float32)
    branch3x3_1_bias = np.zeros((out_3x3,), dtype=np.float32)
    branch3x3_1_stride = 1
    branch3x3_1_padding = 1
    branch3x3_1_dilation = 1
    branch3x3_1_groups = 1
    branch5x5_0_weight = np.zeros((reduce_5x5, in_channels // 1) + _as_tuple(1, 2), dtype=np.float32)
    branch5x5_0_bias = np.zeros((reduce_5x5,), dtype=np.float32)
    branch5x5_0_stride = 1
    branch5x5_0_padding = 0
    branch5x5_0_dilation = 1
    branch5x5_0_groups = 1
    branch5x5_1_weight = np.zeros((out_5x5, reduce_5x5 // 1) + _as_tuple(5, 2), dtype=np.float32)
    branch5x5_1_bias = np.zeros((out_5x5,), dtype=np.float32)
    branch5x5_1_stride = 1
    branch5x5_1_padding = 2
    branch5x5_1_dilation = 1
    branch5x5_1_groups = 1
    branch_pool_0_kernel_size = 3
    branch_pool_0_stride = 1
    branch_pool_0_padding = 1
    branch_pool_1_weight = np.zeros((pool_proj, in_channels // 1) + _as_tuple(1, 2), dtype=np.float32)
    branch_pool_1_bias = np.zeros((pool_proj,), dtype=np.float32)
    branch_pool_1_stride = 1
    branch_pool_1_padding = 0
    branch_pool_1_dilation = 1
    branch_pool_1_groups = 1

def forward(x, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
    branch1x1 = _conv2d(x, branch1x1_weight, branch1x1_bias, branch1x1_stride, branch1x1_padding, branch1x1_dilation, branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, branch3x3_0_weight, branch3x3_0_bias, branch3x3_0_stride, branch3x3_0_padding, branch3x3_0_dilation, branch3x3_0_groups), branch3x3_1_weight, branch3x3_1_bias, branch3x3_1_stride, branch3x3_1_padding, branch3x3_1_dilation, branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, branch5x5_0_weight, branch5x5_0_bias, branch5x5_0_stride, branch5x5_0_padding, branch5x5_0_dilation, branch5x5_0_groups), branch5x5_1_weight, branch5x5_1_bias, branch5x5_1_stride, branch5x5_1_padding, branch5x5_1_dilation, branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, branch_pool_0_kernel_size, branch_pool_0_stride, branch_pool_0_padding), branch_pool_1_weight, branch_pool_1_bias, branch_pool_1_stride, branch_pool_1_padding, branch_pool_1_dilation, branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

