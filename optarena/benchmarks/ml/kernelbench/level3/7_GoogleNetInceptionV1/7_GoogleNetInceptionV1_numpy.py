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

def _inception3a_forward(x):
    branch1x1 = _conv2d(x, inception3a_branch1x1_weight, inception3a_branch1x1_bias, inception3a_branch1x1_stride, inception3a_branch1x1_padding, inception3a_branch1x1_dilation, inception3a_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception3a_branch3x3_0_weight, inception3a_branch3x3_0_bias, inception3a_branch3x3_0_stride, inception3a_branch3x3_0_padding, inception3a_branch3x3_0_dilation, inception3a_branch3x3_0_groups), inception3a_branch3x3_1_weight, inception3a_branch3x3_1_bias, inception3a_branch3x3_1_stride, inception3a_branch3x3_1_padding, inception3a_branch3x3_1_dilation, inception3a_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception3a_branch5x5_0_weight, inception3a_branch5x5_0_bias, inception3a_branch5x5_0_stride, inception3a_branch5x5_0_padding, inception3a_branch5x5_0_dilation, inception3a_branch5x5_0_groups), inception3a_branch5x5_1_weight, inception3a_branch5x5_1_bias, inception3a_branch5x5_1_stride, inception3a_branch5x5_1_padding, inception3a_branch5x5_1_dilation, inception3a_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception3a_branch_pool_0_kernel_size, inception3a_branch_pool_0_stride, inception3a_branch_pool_0_padding), inception3a_branch_pool_1_weight, inception3a_branch_pool_1_bias, inception3a_branch_pool_1_stride, inception3a_branch_pool_1_padding, inception3a_branch_pool_1_dilation, inception3a_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception3b_forward(x):
    branch1x1 = _conv2d(x, inception3b_branch1x1_weight, inception3b_branch1x1_bias, inception3b_branch1x1_stride, inception3b_branch1x1_padding, inception3b_branch1x1_dilation, inception3b_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception3b_branch3x3_0_weight, inception3b_branch3x3_0_bias, inception3b_branch3x3_0_stride, inception3b_branch3x3_0_padding, inception3b_branch3x3_0_dilation, inception3b_branch3x3_0_groups), inception3b_branch3x3_1_weight, inception3b_branch3x3_1_bias, inception3b_branch3x3_1_stride, inception3b_branch3x3_1_padding, inception3b_branch3x3_1_dilation, inception3b_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception3b_branch5x5_0_weight, inception3b_branch5x5_0_bias, inception3b_branch5x5_0_stride, inception3b_branch5x5_0_padding, inception3b_branch5x5_0_dilation, inception3b_branch5x5_0_groups), inception3b_branch5x5_1_weight, inception3b_branch5x5_1_bias, inception3b_branch5x5_1_stride, inception3b_branch5x5_1_padding, inception3b_branch5x5_1_dilation, inception3b_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception3b_branch_pool_0_kernel_size, inception3b_branch_pool_0_stride, inception3b_branch_pool_0_padding), inception3b_branch_pool_1_weight, inception3b_branch_pool_1_bias, inception3b_branch_pool_1_stride, inception3b_branch_pool_1_padding, inception3b_branch_pool_1_dilation, inception3b_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception4a_forward(x):
    branch1x1 = _conv2d(x, inception4a_branch1x1_weight, inception4a_branch1x1_bias, inception4a_branch1x1_stride, inception4a_branch1x1_padding, inception4a_branch1x1_dilation, inception4a_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception4a_branch3x3_0_weight, inception4a_branch3x3_0_bias, inception4a_branch3x3_0_stride, inception4a_branch3x3_0_padding, inception4a_branch3x3_0_dilation, inception4a_branch3x3_0_groups), inception4a_branch3x3_1_weight, inception4a_branch3x3_1_bias, inception4a_branch3x3_1_stride, inception4a_branch3x3_1_padding, inception4a_branch3x3_1_dilation, inception4a_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception4a_branch5x5_0_weight, inception4a_branch5x5_0_bias, inception4a_branch5x5_0_stride, inception4a_branch5x5_0_padding, inception4a_branch5x5_0_dilation, inception4a_branch5x5_0_groups), inception4a_branch5x5_1_weight, inception4a_branch5x5_1_bias, inception4a_branch5x5_1_stride, inception4a_branch5x5_1_padding, inception4a_branch5x5_1_dilation, inception4a_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception4a_branch_pool_0_kernel_size, inception4a_branch_pool_0_stride, inception4a_branch_pool_0_padding), inception4a_branch_pool_1_weight, inception4a_branch_pool_1_bias, inception4a_branch_pool_1_stride, inception4a_branch_pool_1_padding, inception4a_branch_pool_1_dilation, inception4a_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception4b_forward(x):
    branch1x1 = _conv2d(x, inception4b_branch1x1_weight, inception4b_branch1x1_bias, inception4b_branch1x1_stride, inception4b_branch1x1_padding, inception4b_branch1x1_dilation, inception4b_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception4b_branch3x3_0_weight, inception4b_branch3x3_0_bias, inception4b_branch3x3_0_stride, inception4b_branch3x3_0_padding, inception4b_branch3x3_0_dilation, inception4b_branch3x3_0_groups), inception4b_branch3x3_1_weight, inception4b_branch3x3_1_bias, inception4b_branch3x3_1_stride, inception4b_branch3x3_1_padding, inception4b_branch3x3_1_dilation, inception4b_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception4b_branch5x5_0_weight, inception4b_branch5x5_0_bias, inception4b_branch5x5_0_stride, inception4b_branch5x5_0_padding, inception4b_branch5x5_0_dilation, inception4b_branch5x5_0_groups), inception4b_branch5x5_1_weight, inception4b_branch5x5_1_bias, inception4b_branch5x5_1_stride, inception4b_branch5x5_1_padding, inception4b_branch5x5_1_dilation, inception4b_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception4b_branch_pool_0_kernel_size, inception4b_branch_pool_0_stride, inception4b_branch_pool_0_padding), inception4b_branch_pool_1_weight, inception4b_branch_pool_1_bias, inception4b_branch_pool_1_stride, inception4b_branch_pool_1_padding, inception4b_branch_pool_1_dilation, inception4b_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception4c_forward(x):
    branch1x1 = _conv2d(x, inception4c_branch1x1_weight, inception4c_branch1x1_bias, inception4c_branch1x1_stride, inception4c_branch1x1_padding, inception4c_branch1x1_dilation, inception4c_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception4c_branch3x3_0_weight, inception4c_branch3x3_0_bias, inception4c_branch3x3_0_stride, inception4c_branch3x3_0_padding, inception4c_branch3x3_0_dilation, inception4c_branch3x3_0_groups), inception4c_branch3x3_1_weight, inception4c_branch3x3_1_bias, inception4c_branch3x3_1_stride, inception4c_branch3x3_1_padding, inception4c_branch3x3_1_dilation, inception4c_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception4c_branch5x5_0_weight, inception4c_branch5x5_0_bias, inception4c_branch5x5_0_stride, inception4c_branch5x5_0_padding, inception4c_branch5x5_0_dilation, inception4c_branch5x5_0_groups), inception4c_branch5x5_1_weight, inception4c_branch5x5_1_bias, inception4c_branch5x5_1_stride, inception4c_branch5x5_1_padding, inception4c_branch5x5_1_dilation, inception4c_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception4c_branch_pool_0_kernel_size, inception4c_branch_pool_0_stride, inception4c_branch_pool_0_padding), inception4c_branch_pool_1_weight, inception4c_branch_pool_1_bias, inception4c_branch_pool_1_stride, inception4c_branch_pool_1_padding, inception4c_branch_pool_1_dilation, inception4c_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception4d_forward(x):
    branch1x1 = _conv2d(x, inception4d_branch1x1_weight, inception4d_branch1x1_bias, inception4d_branch1x1_stride, inception4d_branch1x1_padding, inception4d_branch1x1_dilation, inception4d_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception4d_branch3x3_0_weight, inception4d_branch3x3_0_bias, inception4d_branch3x3_0_stride, inception4d_branch3x3_0_padding, inception4d_branch3x3_0_dilation, inception4d_branch3x3_0_groups), inception4d_branch3x3_1_weight, inception4d_branch3x3_1_bias, inception4d_branch3x3_1_stride, inception4d_branch3x3_1_padding, inception4d_branch3x3_1_dilation, inception4d_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception4d_branch5x5_0_weight, inception4d_branch5x5_0_bias, inception4d_branch5x5_0_stride, inception4d_branch5x5_0_padding, inception4d_branch5x5_0_dilation, inception4d_branch5x5_0_groups), inception4d_branch5x5_1_weight, inception4d_branch5x5_1_bias, inception4d_branch5x5_1_stride, inception4d_branch5x5_1_padding, inception4d_branch5x5_1_dilation, inception4d_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception4d_branch_pool_0_kernel_size, inception4d_branch_pool_0_stride, inception4d_branch_pool_0_padding), inception4d_branch_pool_1_weight, inception4d_branch_pool_1_bias, inception4d_branch_pool_1_stride, inception4d_branch_pool_1_padding, inception4d_branch_pool_1_dilation, inception4d_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception4e_forward(x):
    branch1x1 = _conv2d(x, inception4e_branch1x1_weight, inception4e_branch1x1_bias, inception4e_branch1x1_stride, inception4e_branch1x1_padding, inception4e_branch1x1_dilation, inception4e_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception4e_branch3x3_0_weight, inception4e_branch3x3_0_bias, inception4e_branch3x3_0_stride, inception4e_branch3x3_0_padding, inception4e_branch3x3_0_dilation, inception4e_branch3x3_0_groups), inception4e_branch3x3_1_weight, inception4e_branch3x3_1_bias, inception4e_branch3x3_1_stride, inception4e_branch3x3_1_padding, inception4e_branch3x3_1_dilation, inception4e_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception4e_branch5x5_0_weight, inception4e_branch5x5_0_bias, inception4e_branch5x5_0_stride, inception4e_branch5x5_0_padding, inception4e_branch5x5_0_dilation, inception4e_branch5x5_0_groups), inception4e_branch5x5_1_weight, inception4e_branch5x5_1_bias, inception4e_branch5x5_1_stride, inception4e_branch5x5_1_padding, inception4e_branch5x5_1_dilation, inception4e_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception4e_branch_pool_0_kernel_size, inception4e_branch_pool_0_stride, inception4e_branch_pool_0_padding), inception4e_branch_pool_1_weight, inception4e_branch_pool_1_bias, inception4e_branch_pool_1_stride, inception4e_branch_pool_1_padding, inception4e_branch_pool_1_dilation, inception4e_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception5a_forward(x):
    branch1x1 = _conv2d(x, inception5a_branch1x1_weight, inception5a_branch1x1_bias, inception5a_branch1x1_stride, inception5a_branch1x1_padding, inception5a_branch1x1_dilation, inception5a_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception5a_branch3x3_0_weight, inception5a_branch3x3_0_bias, inception5a_branch3x3_0_stride, inception5a_branch3x3_0_padding, inception5a_branch3x3_0_dilation, inception5a_branch3x3_0_groups), inception5a_branch3x3_1_weight, inception5a_branch3x3_1_bias, inception5a_branch3x3_1_stride, inception5a_branch3x3_1_padding, inception5a_branch3x3_1_dilation, inception5a_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception5a_branch5x5_0_weight, inception5a_branch5x5_0_bias, inception5a_branch5x5_0_stride, inception5a_branch5x5_0_padding, inception5a_branch5x5_0_dilation, inception5a_branch5x5_0_groups), inception5a_branch5x5_1_weight, inception5a_branch5x5_1_bias, inception5a_branch5x5_1_stride, inception5a_branch5x5_1_padding, inception5a_branch5x5_1_dilation, inception5a_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception5a_branch_pool_0_kernel_size, inception5a_branch_pool_0_stride, inception5a_branch_pool_0_padding), inception5a_branch_pool_1_weight, inception5a_branch_pool_1_bias, inception5a_branch_pool_1_stride, inception5a_branch_pool_1_padding, inception5a_branch_pool_1_dilation, inception5a_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def _inception5b_forward(x):
    branch1x1 = _conv2d(x, inception5b_branch1x1_weight, inception5b_branch1x1_bias, inception5b_branch1x1_stride, inception5b_branch1x1_padding, inception5b_branch1x1_dilation, inception5b_branch1x1_groups)
    branch3x3 = _conv2d(_conv2d(x, inception5b_branch3x3_0_weight, inception5b_branch3x3_0_bias, inception5b_branch3x3_0_stride, inception5b_branch3x3_0_padding, inception5b_branch3x3_0_dilation, inception5b_branch3x3_0_groups), inception5b_branch3x3_1_weight, inception5b_branch3x3_1_bias, inception5b_branch3x3_1_stride, inception5b_branch3x3_1_padding, inception5b_branch3x3_1_dilation, inception5b_branch3x3_1_groups)
    branch5x5 = _conv2d(_conv2d(x, inception5b_branch5x5_0_weight, inception5b_branch5x5_0_bias, inception5b_branch5x5_0_stride, inception5b_branch5x5_0_padding, inception5b_branch5x5_0_dilation, inception5b_branch5x5_0_groups), inception5b_branch5x5_1_weight, inception5b_branch5x5_1_bias, inception5b_branch5x5_1_stride, inception5b_branch5x5_1_padding, inception5b_branch5x5_1_dilation, inception5b_branch5x5_1_groups)
    branch_pool = _conv2d(_maxpool2d(x, inception5b_branch_pool_0_kernel_size, inception5b_branch_pool_0_stride, inception5b_branch_pool_0_padding), inception5b_branch_pool_1_weight, inception5b_branch_pool_1_bias, inception5b_branch_pool_1_stride, inception5b_branch_pool_1_padding, inception5b_branch_pool_1_dilation, inception5b_branch_pool_1_groups)
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return np.concatenate(outputs, axis=1)

def init(num_classes=1000):
    global conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, maxpool1_kernel_size, maxpool1_stride, maxpool1_padding, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups, conv3_weight, conv3_bias, conv3_stride, conv3_padding, conv3_dilation, conv3_groups, maxpool2_kernel_size, maxpool2_stride, maxpool2_padding, inception3a_branch1x1_weight, inception3a_branch1x1_bias, inception3a_branch1x1_stride, inception3a_branch1x1_padding, inception3a_branch1x1_dilation, inception3a_branch1x1_groups, inception3a_branch3x3_0_weight, inception3a_branch3x3_0_bias, inception3a_branch3x3_0_stride, inception3a_branch3x3_0_padding, inception3a_branch3x3_0_dilation, inception3a_branch3x3_0_groups, inception3a_branch3x3_1_weight, inception3a_branch3x3_1_bias, inception3a_branch3x3_1_stride, inception3a_branch3x3_1_padding, inception3a_branch3x3_1_dilation, inception3a_branch3x3_1_groups, inception3a_branch5x5_0_weight, inception3a_branch5x5_0_bias, inception3a_branch5x5_0_stride, inception3a_branch5x5_0_padding, inception3a_branch5x5_0_dilation, inception3a_branch5x5_0_groups, inception3a_branch5x5_1_weight, inception3a_branch5x5_1_bias, inception3a_branch5x5_1_stride, inception3a_branch5x5_1_padding, inception3a_branch5x5_1_dilation, inception3a_branch5x5_1_groups, inception3a_branch_pool_0_kernel_size, inception3a_branch_pool_0_stride, inception3a_branch_pool_0_padding, inception3a_branch_pool_1_weight, inception3a_branch_pool_1_bias, inception3a_branch_pool_1_stride, inception3a_branch_pool_1_padding, inception3a_branch_pool_1_dilation, inception3a_branch_pool_1_groups, inception3b_branch1x1_weight, inception3b_branch1x1_bias, inception3b_branch1x1_stride, inception3b_branch1x1_padding, inception3b_branch1x1_dilation, inception3b_branch1x1_groups, inception3b_branch3x3_0_weight, inception3b_branch3x3_0_bias, inception3b_branch3x3_0_stride, inception3b_branch3x3_0_padding, inception3b_branch3x3_0_dilation, inception3b_branch3x3_0_groups, inception3b_branch3x3_1_weight, inception3b_branch3x3_1_bias, inception3b_branch3x3_1_stride, inception3b_branch3x3_1_padding, inception3b_branch3x3_1_dilation, inception3b_branch3x3_1_groups, inception3b_branch5x5_0_weight, inception3b_branch5x5_0_bias, inception3b_branch5x5_0_stride, inception3b_branch5x5_0_padding, inception3b_branch5x5_0_dilation, inception3b_branch5x5_0_groups, inception3b_branch5x5_1_weight, inception3b_branch5x5_1_bias, inception3b_branch5x5_1_stride, inception3b_branch5x5_1_padding, inception3b_branch5x5_1_dilation, inception3b_branch5x5_1_groups, inception3b_branch_pool_0_kernel_size, inception3b_branch_pool_0_stride, inception3b_branch_pool_0_padding, inception3b_branch_pool_1_weight, inception3b_branch_pool_1_bias, inception3b_branch_pool_1_stride, inception3b_branch_pool_1_padding, inception3b_branch_pool_1_dilation, inception3b_branch_pool_1_groups, maxpool3_kernel_size, maxpool3_stride, maxpool3_padding, inception4a_branch1x1_weight, inception4a_branch1x1_bias, inception4a_branch1x1_stride, inception4a_branch1x1_padding, inception4a_branch1x1_dilation, inception4a_branch1x1_groups, inception4a_branch3x3_0_weight, inception4a_branch3x3_0_bias, inception4a_branch3x3_0_stride, inception4a_branch3x3_0_padding, inception4a_branch3x3_0_dilation, inception4a_branch3x3_0_groups, inception4a_branch3x3_1_weight, inception4a_branch3x3_1_bias, inception4a_branch3x3_1_stride, inception4a_branch3x3_1_padding, inception4a_branch3x3_1_dilation, inception4a_branch3x3_1_groups, inception4a_branch5x5_0_weight, inception4a_branch5x5_0_bias, inception4a_branch5x5_0_stride, inception4a_branch5x5_0_padding, inception4a_branch5x5_0_dilation, inception4a_branch5x5_0_groups, inception4a_branch5x5_1_weight, inception4a_branch5x5_1_bias, inception4a_branch5x5_1_stride, inception4a_branch5x5_1_padding, inception4a_branch5x5_1_dilation, inception4a_branch5x5_1_groups, inception4a_branch_pool_0_kernel_size, inception4a_branch_pool_0_stride, inception4a_branch_pool_0_padding, inception4a_branch_pool_1_weight, inception4a_branch_pool_1_bias, inception4a_branch_pool_1_stride, inception4a_branch_pool_1_padding, inception4a_branch_pool_1_dilation, inception4a_branch_pool_1_groups, inception4b_branch1x1_weight, inception4b_branch1x1_bias, inception4b_branch1x1_stride, inception4b_branch1x1_padding, inception4b_branch1x1_dilation, inception4b_branch1x1_groups, inception4b_branch3x3_0_weight, inception4b_branch3x3_0_bias, inception4b_branch3x3_0_stride, inception4b_branch3x3_0_padding, inception4b_branch3x3_0_dilation, inception4b_branch3x3_0_groups, inception4b_branch3x3_1_weight, inception4b_branch3x3_1_bias, inception4b_branch3x3_1_stride, inception4b_branch3x3_1_padding, inception4b_branch3x3_1_dilation, inception4b_branch3x3_1_groups, inception4b_branch5x5_0_weight, inception4b_branch5x5_0_bias, inception4b_branch5x5_0_stride, inception4b_branch5x5_0_padding, inception4b_branch5x5_0_dilation, inception4b_branch5x5_0_groups, inception4b_branch5x5_1_weight, inception4b_branch5x5_1_bias, inception4b_branch5x5_1_stride, inception4b_branch5x5_1_padding, inception4b_branch5x5_1_dilation, inception4b_branch5x5_1_groups, inception4b_branch_pool_0_kernel_size, inception4b_branch_pool_0_stride, inception4b_branch_pool_0_padding, inception4b_branch_pool_1_weight, inception4b_branch_pool_1_bias, inception4b_branch_pool_1_stride, inception4b_branch_pool_1_padding, inception4b_branch_pool_1_dilation, inception4b_branch_pool_1_groups, inception4c_branch1x1_weight, inception4c_branch1x1_bias, inception4c_branch1x1_stride, inception4c_branch1x1_padding, inception4c_branch1x1_dilation, inception4c_branch1x1_groups, inception4c_branch3x3_0_weight, inception4c_branch3x3_0_bias, inception4c_branch3x3_0_stride, inception4c_branch3x3_0_padding, inception4c_branch3x3_0_dilation, inception4c_branch3x3_0_groups, inception4c_branch3x3_1_weight, inception4c_branch3x3_1_bias, inception4c_branch3x3_1_stride, inception4c_branch3x3_1_padding, inception4c_branch3x3_1_dilation, inception4c_branch3x3_1_groups, inception4c_branch5x5_0_weight, inception4c_branch5x5_0_bias, inception4c_branch5x5_0_stride, inception4c_branch5x5_0_padding, inception4c_branch5x5_0_dilation, inception4c_branch5x5_0_groups, inception4c_branch5x5_1_weight, inception4c_branch5x5_1_bias, inception4c_branch5x5_1_stride, inception4c_branch5x5_1_padding, inception4c_branch5x5_1_dilation, inception4c_branch5x5_1_groups, inception4c_branch_pool_0_kernel_size, inception4c_branch_pool_0_stride, inception4c_branch_pool_0_padding, inception4c_branch_pool_1_weight, inception4c_branch_pool_1_bias, inception4c_branch_pool_1_stride, inception4c_branch_pool_1_padding, inception4c_branch_pool_1_dilation, inception4c_branch_pool_1_groups, inception4d_branch1x1_weight, inception4d_branch1x1_bias, inception4d_branch1x1_stride, inception4d_branch1x1_padding, inception4d_branch1x1_dilation, inception4d_branch1x1_groups, inception4d_branch3x3_0_weight, inception4d_branch3x3_0_bias, inception4d_branch3x3_0_stride, inception4d_branch3x3_0_padding, inception4d_branch3x3_0_dilation, inception4d_branch3x3_0_groups, inception4d_branch3x3_1_weight, inception4d_branch3x3_1_bias, inception4d_branch3x3_1_stride, inception4d_branch3x3_1_padding, inception4d_branch3x3_1_dilation, inception4d_branch3x3_1_groups, inception4d_branch5x5_0_weight, inception4d_branch5x5_0_bias, inception4d_branch5x5_0_stride, inception4d_branch5x5_0_padding, inception4d_branch5x5_0_dilation, inception4d_branch5x5_0_groups, inception4d_branch5x5_1_weight, inception4d_branch5x5_1_bias, inception4d_branch5x5_1_stride, inception4d_branch5x5_1_padding, inception4d_branch5x5_1_dilation, inception4d_branch5x5_1_groups, inception4d_branch_pool_0_kernel_size, inception4d_branch_pool_0_stride, inception4d_branch_pool_0_padding, inception4d_branch_pool_1_weight, inception4d_branch_pool_1_bias, inception4d_branch_pool_1_stride, inception4d_branch_pool_1_padding, inception4d_branch_pool_1_dilation, inception4d_branch_pool_1_groups, inception4e_branch1x1_weight, inception4e_branch1x1_bias, inception4e_branch1x1_stride, inception4e_branch1x1_padding, inception4e_branch1x1_dilation, inception4e_branch1x1_groups, inception4e_branch3x3_0_weight, inception4e_branch3x3_0_bias, inception4e_branch3x3_0_stride, inception4e_branch3x3_0_padding, inception4e_branch3x3_0_dilation, inception4e_branch3x3_0_groups, inception4e_branch3x3_1_weight, inception4e_branch3x3_1_bias, inception4e_branch3x3_1_stride, inception4e_branch3x3_1_padding, inception4e_branch3x3_1_dilation, inception4e_branch3x3_1_groups, inception4e_branch5x5_0_weight, inception4e_branch5x5_0_bias, inception4e_branch5x5_0_stride, inception4e_branch5x5_0_padding, inception4e_branch5x5_0_dilation, inception4e_branch5x5_0_groups, inception4e_branch5x5_1_weight, inception4e_branch5x5_1_bias, inception4e_branch5x5_1_stride, inception4e_branch5x5_1_padding, inception4e_branch5x5_1_dilation, inception4e_branch5x5_1_groups, inception4e_branch_pool_0_kernel_size, inception4e_branch_pool_0_stride, inception4e_branch_pool_0_padding, inception4e_branch_pool_1_weight, inception4e_branch_pool_1_bias, inception4e_branch_pool_1_stride, inception4e_branch_pool_1_padding, inception4e_branch_pool_1_dilation, inception4e_branch_pool_1_groups, maxpool4_kernel_size, maxpool4_stride, maxpool4_padding, inception5a_branch1x1_weight, inception5a_branch1x1_bias, inception5a_branch1x1_stride, inception5a_branch1x1_padding, inception5a_branch1x1_dilation, inception5a_branch1x1_groups, inception5a_branch3x3_0_weight, inception5a_branch3x3_0_bias, inception5a_branch3x3_0_stride, inception5a_branch3x3_0_padding, inception5a_branch3x3_0_dilation, inception5a_branch3x3_0_groups, inception5a_branch3x3_1_weight, inception5a_branch3x3_1_bias, inception5a_branch3x3_1_stride, inception5a_branch3x3_1_padding, inception5a_branch3x3_1_dilation, inception5a_branch3x3_1_groups, inception5a_branch5x5_0_weight, inception5a_branch5x5_0_bias, inception5a_branch5x5_0_stride, inception5a_branch5x5_0_padding, inception5a_branch5x5_0_dilation, inception5a_branch5x5_0_groups, inception5a_branch5x5_1_weight, inception5a_branch5x5_1_bias, inception5a_branch5x5_1_stride, inception5a_branch5x5_1_padding, inception5a_branch5x5_1_dilation, inception5a_branch5x5_1_groups, inception5a_branch_pool_0_kernel_size, inception5a_branch_pool_0_stride, inception5a_branch_pool_0_padding, inception5a_branch_pool_1_weight, inception5a_branch_pool_1_bias, inception5a_branch_pool_1_stride, inception5a_branch_pool_1_padding, inception5a_branch_pool_1_dilation, inception5a_branch_pool_1_groups, inception5b_branch1x1_weight, inception5b_branch1x1_bias, inception5b_branch1x1_stride, inception5b_branch1x1_padding, inception5b_branch1x1_dilation, inception5b_branch1x1_groups, inception5b_branch3x3_0_weight, inception5b_branch3x3_0_bias, inception5b_branch3x3_0_stride, inception5b_branch3x3_0_padding, inception5b_branch3x3_0_dilation, inception5b_branch3x3_0_groups, inception5b_branch3x3_1_weight, inception5b_branch3x3_1_bias, inception5b_branch3x3_1_stride, inception5b_branch3x3_1_padding, inception5b_branch3x3_1_dilation, inception5b_branch3x3_1_groups, inception5b_branch5x5_0_weight, inception5b_branch5x5_0_bias, inception5b_branch5x5_0_stride, inception5b_branch5x5_0_padding, inception5b_branch5x5_0_dilation, inception5b_branch5x5_0_groups, inception5b_branch5x5_1_weight, inception5b_branch5x5_1_bias, inception5b_branch5x5_1_stride, inception5b_branch5x5_1_padding, inception5b_branch5x5_1_dilation, inception5b_branch5x5_1_groups, inception5b_branch_pool_0_kernel_size, inception5b_branch_pool_0_stride, inception5b_branch_pool_0_padding, inception5b_branch_pool_1_weight, inception5b_branch_pool_1_bias, inception5b_branch_pool_1_stride, inception5b_branch_pool_1_padding, inception5b_branch_pool_1_dilation, inception5b_branch_pool_1_groups, avgpool_output_size, dropout, fc_weight, fc_bias
    conv1_weight = np.zeros((64, 3 // 1) + _as_tuple(7, 2), dtype=np.float32)
    conv1_bias = np.zeros((64,), dtype=np.float32)
    conv1_stride = 2
    conv1_padding = 3
    conv1_dilation = 1
    conv1_groups = 1
    maxpool1_kernel_size = 3
    maxpool1_stride = 2
    maxpool1_padding = 1
    conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(1, 2), dtype=np.float32)
    conv2_bias = np.zeros((64,), dtype=np.float32)
    conv2_stride = 1
    conv2_padding = 0
    conv2_dilation = 1
    conv2_groups = 1
    conv3_weight = np.zeros((192, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    conv3_bias = np.zeros((192,), dtype=np.float32)
    conv3_stride = 1
    conv3_padding = 1
    conv3_dilation = 1
    conv3_groups = 1
    maxpool2_kernel_size = 3
    maxpool2_stride = 2
    maxpool2_padding = 1
    inception3a_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3a_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception3a_branch1x1_stride = 1
    inception3a_branch1x1_padding = 0
    inception3a_branch1x1_dilation = 1
    inception3a_branch1x1_groups = 1
    inception3a_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3a_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception3a_branch3x3_0_stride = 1
    inception3a_branch3x3_0_padding = 0
    inception3a_branch3x3_0_dilation = 1
    inception3a_branch3x3_0_groups = 1
    inception3a_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception3a_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception3a_branch3x3_1_stride = 1
    inception3a_branch3x3_1_padding = 1
    inception3a_branch3x3_1_dilation = 1
    inception3a_branch3x3_1_groups = 1
    inception3a_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3a_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception3a_branch5x5_0_stride = 1
    inception3a_branch5x5_0_padding = 0
    inception3a_branch5x5_0_dilation = 1
    inception3a_branch5x5_0_groups = 1
    inception3a_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception3a_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception3a_branch5x5_1_stride = 1
    inception3a_branch5x5_1_padding = 2
    inception3a_branch5x5_1_dilation = 1
    inception3a_branch5x5_1_groups = 1
    inception3a_branch_pool_0_kernel_size = 3
    inception3a_branch_pool_0_stride = 1
    inception3a_branch_pool_0_padding = 1
    inception3a_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3a_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception3a_branch_pool_1_stride = 1
    inception3a_branch_pool_1_padding = 0
    inception3a_branch_pool_1_dilation = 1
    inception3a_branch_pool_1_groups = 1
    inception3b_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3b_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception3b_branch1x1_stride = 1
    inception3b_branch1x1_padding = 0
    inception3b_branch1x1_dilation = 1
    inception3b_branch1x1_groups = 1
    inception3b_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3b_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception3b_branch3x3_0_stride = 1
    inception3b_branch3x3_0_padding = 0
    inception3b_branch3x3_0_dilation = 1
    inception3b_branch3x3_0_groups = 1
    inception3b_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception3b_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception3b_branch3x3_1_stride = 1
    inception3b_branch3x3_1_padding = 1
    inception3b_branch3x3_1_dilation = 1
    inception3b_branch3x3_1_groups = 1
    inception3b_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3b_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception3b_branch5x5_0_stride = 1
    inception3b_branch5x5_0_padding = 0
    inception3b_branch5x5_0_dilation = 1
    inception3b_branch5x5_0_groups = 1
    inception3b_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception3b_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception3b_branch5x5_1_stride = 1
    inception3b_branch5x5_1_padding = 2
    inception3b_branch5x5_1_dilation = 1
    inception3b_branch5x5_1_groups = 1
    inception3b_branch_pool_0_kernel_size = 3
    inception3b_branch_pool_0_stride = 1
    inception3b_branch_pool_0_padding = 1
    inception3b_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception3b_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception3b_branch_pool_1_stride = 1
    inception3b_branch_pool_1_padding = 0
    inception3b_branch_pool_1_dilation = 1
    inception3b_branch_pool_1_groups = 1
    maxpool3_kernel_size = 3
    maxpool3_stride = 2
    maxpool3_padding = 1
    inception4a_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4a_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception4a_branch1x1_stride = 1
    inception4a_branch1x1_padding = 0
    inception4a_branch1x1_dilation = 1
    inception4a_branch1x1_groups = 1
    inception4a_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4a_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception4a_branch3x3_0_stride = 1
    inception4a_branch3x3_0_padding = 0
    inception4a_branch3x3_0_dilation = 1
    inception4a_branch3x3_0_groups = 1
    inception4a_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception4a_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception4a_branch3x3_1_stride = 1
    inception4a_branch3x3_1_padding = 1
    inception4a_branch3x3_1_dilation = 1
    inception4a_branch3x3_1_groups = 1
    inception4a_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4a_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception4a_branch5x5_0_stride = 1
    inception4a_branch5x5_0_padding = 0
    inception4a_branch5x5_0_dilation = 1
    inception4a_branch5x5_0_groups = 1
    inception4a_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception4a_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception4a_branch5x5_1_stride = 1
    inception4a_branch5x5_1_padding = 2
    inception4a_branch5x5_1_dilation = 1
    inception4a_branch5x5_1_groups = 1
    inception4a_branch_pool_0_kernel_size = 3
    inception4a_branch_pool_0_stride = 1
    inception4a_branch_pool_0_padding = 1
    inception4a_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4a_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception4a_branch_pool_1_stride = 1
    inception4a_branch_pool_1_padding = 0
    inception4a_branch_pool_1_dilation = 1
    inception4a_branch_pool_1_groups = 1
    inception4b_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4b_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception4b_branch1x1_stride = 1
    inception4b_branch1x1_padding = 0
    inception4b_branch1x1_dilation = 1
    inception4b_branch1x1_groups = 1
    inception4b_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4b_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception4b_branch3x3_0_stride = 1
    inception4b_branch3x3_0_padding = 0
    inception4b_branch3x3_0_dilation = 1
    inception4b_branch3x3_0_groups = 1
    inception4b_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception4b_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception4b_branch3x3_1_stride = 1
    inception4b_branch3x3_1_padding = 1
    inception4b_branch3x3_1_dilation = 1
    inception4b_branch3x3_1_groups = 1
    inception4b_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4b_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception4b_branch5x5_0_stride = 1
    inception4b_branch5x5_0_padding = 0
    inception4b_branch5x5_0_dilation = 1
    inception4b_branch5x5_0_groups = 1
    inception4b_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception4b_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception4b_branch5x5_1_stride = 1
    inception4b_branch5x5_1_padding = 2
    inception4b_branch5x5_1_dilation = 1
    inception4b_branch5x5_1_groups = 1
    inception4b_branch_pool_0_kernel_size = 3
    inception4b_branch_pool_0_stride = 1
    inception4b_branch_pool_0_padding = 1
    inception4b_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4b_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception4b_branch_pool_1_stride = 1
    inception4b_branch_pool_1_padding = 0
    inception4b_branch_pool_1_dilation = 1
    inception4b_branch_pool_1_groups = 1
    inception4c_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4c_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception4c_branch1x1_stride = 1
    inception4c_branch1x1_padding = 0
    inception4c_branch1x1_dilation = 1
    inception4c_branch1x1_groups = 1
    inception4c_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4c_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception4c_branch3x3_0_stride = 1
    inception4c_branch3x3_0_padding = 0
    inception4c_branch3x3_0_dilation = 1
    inception4c_branch3x3_0_groups = 1
    inception4c_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception4c_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception4c_branch3x3_1_stride = 1
    inception4c_branch3x3_1_padding = 1
    inception4c_branch3x3_1_dilation = 1
    inception4c_branch3x3_1_groups = 1
    inception4c_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4c_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception4c_branch5x5_0_stride = 1
    inception4c_branch5x5_0_padding = 0
    inception4c_branch5x5_0_dilation = 1
    inception4c_branch5x5_0_groups = 1
    inception4c_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception4c_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception4c_branch5x5_1_stride = 1
    inception4c_branch5x5_1_padding = 2
    inception4c_branch5x5_1_dilation = 1
    inception4c_branch5x5_1_groups = 1
    inception4c_branch_pool_0_kernel_size = 3
    inception4c_branch_pool_0_stride = 1
    inception4c_branch_pool_0_padding = 1
    inception4c_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4c_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception4c_branch_pool_1_stride = 1
    inception4c_branch_pool_1_padding = 0
    inception4c_branch_pool_1_dilation = 1
    inception4c_branch_pool_1_groups = 1
    inception4d_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4d_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception4d_branch1x1_stride = 1
    inception4d_branch1x1_padding = 0
    inception4d_branch1x1_dilation = 1
    inception4d_branch1x1_groups = 1
    inception4d_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4d_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception4d_branch3x3_0_stride = 1
    inception4d_branch3x3_0_padding = 0
    inception4d_branch3x3_0_dilation = 1
    inception4d_branch3x3_0_groups = 1
    inception4d_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception4d_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception4d_branch3x3_1_stride = 1
    inception4d_branch3x3_1_padding = 1
    inception4d_branch3x3_1_dilation = 1
    inception4d_branch3x3_1_groups = 1
    inception4d_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4d_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception4d_branch5x5_0_stride = 1
    inception4d_branch5x5_0_padding = 0
    inception4d_branch5x5_0_dilation = 1
    inception4d_branch5x5_0_groups = 1
    inception4d_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception4d_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception4d_branch5x5_1_stride = 1
    inception4d_branch5x5_1_padding = 2
    inception4d_branch5x5_1_dilation = 1
    inception4d_branch5x5_1_groups = 1
    inception4d_branch_pool_0_kernel_size = 3
    inception4d_branch_pool_0_stride = 1
    inception4d_branch_pool_0_padding = 1
    inception4d_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4d_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception4d_branch_pool_1_stride = 1
    inception4d_branch_pool_1_padding = 0
    inception4d_branch_pool_1_dilation = 1
    inception4d_branch_pool_1_groups = 1
    inception4e_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4e_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception4e_branch1x1_stride = 1
    inception4e_branch1x1_padding = 0
    inception4e_branch1x1_dilation = 1
    inception4e_branch1x1_groups = 1
    inception4e_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4e_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception4e_branch3x3_0_stride = 1
    inception4e_branch3x3_0_padding = 0
    inception4e_branch3x3_0_dilation = 1
    inception4e_branch3x3_0_groups = 1
    inception4e_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception4e_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception4e_branch3x3_1_stride = 1
    inception4e_branch3x3_1_padding = 1
    inception4e_branch3x3_1_dilation = 1
    inception4e_branch3x3_1_groups = 1
    inception4e_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4e_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception4e_branch5x5_0_stride = 1
    inception4e_branch5x5_0_padding = 0
    inception4e_branch5x5_0_dilation = 1
    inception4e_branch5x5_0_groups = 1
    inception4e_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception4e_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception4e_branch5x5_1_stride = 1
    inception4e_branch5x5_1_padding = 2
    inception4e_branch5x5_1_dilation = 1
    inception4e_branch5x5_1_groups = 1
    inception4e_branch_pool_0_kernel_size = 3
    inception4e_branch_pool_0_stride = 1
    inception4e_branch_pool_0_padding = 1
    inception4e_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception4e_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception4e_branch_pool_1_stride = 1
    inception4e_branch_pool_1_padding = 0
    inception4e_branch_pool_1_dilation = 1
    inception4e_branch_pool_1_groups = 1
    maxpool4_kernel_size = 3
    maxpool4_stride = 2
    maxpool4_padding = 1
    inception5a_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5a_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception5a_branch1x1_stride = 1
    inception5a_branch1x1_padding = 0
    inception5a_branch1x1_dilation = 1
    inception5a_branch1x1_groups = 1
    inception5a_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5a_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception5a_branch3x3_0_stride = 1
    inception5a_branch3x3_0_padding = 0
    inception5a_branch3x3_0_dilation = 1
    inception5a_branch3x3_0_groups = 1
    inception5a_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception5a_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception5a_branch3x3_1_stride = 1
    inception5a_branch3x3_1_padding = 1
    inception5a_branch3x3_1_dilation = 1
    inception5a_branch3x3_1_groups = 1
    inception5a_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5a_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception5a_branch5x5_0_stride = 1
    inception5a_branch5x5_0_padding = 0
    inception5a_branch5x5_0_dilation = 1
    inception5a_branch5x5_0_groups = 1
    inception5a_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception5a_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception5a_branch5x5_1_stride = 1
    inception5a_branch5x5_1_padding = 2
    inception5a_branch5x5_1_dilation = 1
    inception5a_branch5x5_1_groups = 1
    inception5a_branch_pool_0_kernel_size = 3
    inception5a_branch_pool_0_stride = 1
    inception5a_branch_pool_0_padding = 1
    inception5a_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5a_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception5a_branch_pool_1_stride = 1
    inception5a_branch_pool_1_padding = 0
    inception5a_branch_pool_1_dilation = 1
    inception5a_branch_pool_1_groups = 1
    inception5b_branch1x1_weight = np.zeros((64, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5b_branch1x1_bias = np.zeros((64,), dtype=np.float32)
    inception5b_branch1x1_stride = 1
    inception5b_branch1x1_padding = 0
    inception5b_branch1x1_dilation = 1
    inception5b_branch1x1_groups = 1
    inception5b_branch3x3_0_weight = np.zeros((96, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5b_branch3x3_0_bias = np.zeros((96,), dtype=np.float32)
    inception5b_branch3x3_0_stride = 1
    inception5b_branch3x3_0_padding = 0
    inception5b_branch3x3_0_dilation = 1
    inception5b_branch3x3_0_groups = 1
    inception5b_branch3x3_1_weight = np.zeros((128, 96 // 1) + _as_tuple(3, 2), dtype=np.float32)
    inception5b_branch3x3_1_bias = np.zeros((128,), dtype=np.float32)
    inception5b_branch3x3_1_stride = 1
    inception5b_branch3x3_1_padding = 1
    inception5b_branch3x3_1_dilation = 1
    inception5b_branch3x3_1_groups = 1
    inception5b_branch5x5_0_weight = np.zeros((16, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5b_branch5x5_0_bias = np.zeros((16,), dtype=np.float32)
    inception5b_branch5x5_0_stride = 1
    inception5b_branch5x5_0_padding = 0
    inception5b_branch5x5_0_dilation = 1
    inception5b_branch5x5_0_groups = 1
    inception5b_branch5x5_1_weight = np.zeros((32, 16 // 1) + _as_tuple(5, 2), dtype=np.float32)
    inception5b_branch5x5_1_bias = np.zeros((32,), dtype=np.float32)
    inception5b_branch5x5_1_stride = 1
    inception5b_branch5x5_1_padding = 2
    inception5b_branch5x5_1_dilation = 1
    inception5b_branch5x5_1_groups = 1
    inception5b_branch_pool_0_kernel_size = 3
    inception5b_branch_pool_0_stride = 1
    inception5b_branch_pool_0_padding = 1
    inception5b_branch_pool_1_weight = np.zeros((32, 192 // 1) + _as_tuple(1, 2), dtype=np.float32)
    inception5b_branch_pool_1_bias = np.zeros((32,), dtype=np.float32)
    inception5b_branch_pool_1_stride = 1
    inception5b_branch_pool_1_padding = 0
    inception5b_branch_pool_1_dilation = 1
    inception5b_branch_pool_1_groups = 1
    avgpool_output_size = (1, 1)
    dropout = None
    fc_weight = np.zeros((num_classes, 1024), dtype=np.float32)
    fc_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000):
    x = _maxpool2d(np.maximum(_conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups), 0), maxpool1_kernel_size, maxpool1_stride, maxpool1_padding)
    x = np.maximum(_conv2d(x, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups), 0)
    x = _maxpool2d(np.maximum(_conv2d(x, conv3_weight, conv3_bias, conv3_stride, conv3_padding, conv3_dilation, conv3_groups), 0), maxpool2_kernel_size, maxpool2_stride, maxpool2_padding)
    x = _inception3a_forward(x)
    x = _inception3b_forward(x)
    x = _maxpool2d(x, maxpool3_kernel_size, maxpool3_stride, maxpool3_padding)
    x = _inception4a_forward(x)
    x = _inception4b_forward(x)
    x = _inception4c_forward(x)
    x = _inception4d_forward(x)
    x = _inception4e_forward(x)
    x = _maxpool2d(x, maxpool4_kernel_size, maxpool4_stride, maxpool4_padding)
    x = _inception5a_forward(x)
    x = _inception5b_forward(x)
    x = _adaptive_avg_pool2d(x, avgpool_output_size)
    x = np.reshape(x, (x.shape[0], -1))
    x = x
    x = ((x) @ fc_weight.T + fc_bias)
    return x

