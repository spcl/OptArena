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

def init(num_classes=1000):
    global conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, relu1, maxpool1_kernel_size, maxpool1_stride, maxpool1_padding, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups, relu2, maxpool2_kernel_size, maxpool2_stride, maxpool2_padding, conv3_weight, conv3_bias, conv3_stride, conv3_padding, conv3_dilation, conv3_groups, relu3, conv4_weight, conv4_bias, conv4_stride, conv4_padding, conv4_dilation, conv4_groups, relu4, conv5_weight, conv5_bias, conv5_stride, conv5_padding, conv5_dilation, conv5_groups, relu5, maxpool3_kernel_size, maxpool3_stride, maxpool3_padding, fc1_weight, fc1_bias, relu6, dropout1, fc2_weight, fc2_bias, relu7, dropout2, fc3_weight, fc3_bias
    conv1_weight = np.zeros((96, 3 // 1) + _as_tuple(11, 2), dtype=np.float32)
    conv1_bias = np.zeros((96,), dtype=np.float32)
    conv1_stride = 4
    conv1_padding = 2
    conv1_dilation = 1
    conv1_groups = 1
    relu1 = None
    maxpool1_kernel_size = 3
    maxpool1_stride = 2
    maxpool1_padding = 0
    conv2_weight = np.zeros((256, 96 // 1) + _as_tuple(5, 2), dtype=np.float32)
    conv2_bias = np.zeros((256,), dtype=np.float32)
    conv2_stride = 1
    conv2_padding = 2
    conv2_dilation = 1
    conv2_groups = 1
    relu2 = None
    maxpool2_kernel_size = 3
    maxpool2_stride = 2
    maxpool2_padding = 0
    conv3_weight = np.zeros((384, 256 // 1) + _as_tuple(3, 2), dtype=np.float32)
    conv3_bias = np.zeros((384,), dtype=np.float32)
    conv3_stride = 1
    conv3_padding = 1
    conv3_dilation = 1
    conv3_groups = 1
    relu3 = None
    conv4_weight = np.zeros((384, 384 // 1) + _as_tuple(3, 2), dtype=np.float32)
    conv4_bias = np.zeros((384,), dtype=np.float32)
    conv4_stride = 1
    conv4_padding = 1
    conv4_dilation = 1
    conv4_groups = 1
    relu4 = None
    conv5_weight = np.zeros((256, 384 // 1) + _as_tuple(3, 2), dtype=np.float32)
    conv5_bias = np.zeros((256,), dtype=np.float32)
    conv5_stride = 1
    conv5_padding = 1
    conv5_dilation = 1
    conv5_groups = 1
    relu5 = None
    maxpool3_kernel_size = 3
    maxpool3_stride = 2
    maxpool3_padding = 0
    fc1_weight = np.zeros((4096, 256 * 6 * 6), dtype=np.float32)
    fc1_bias = np.zeros((4096,), dtype=np.float32) if True else np.zeros((4096,), dtype=np.float32)
    relu6 = None
    dropout1 = None
    fc2_weight = np.zeros((4096, 4096), dtype=np.float32)
    fc2_bias = np.zeros((4096,), dtype=np.float32) if True else np.zeros((4096,), dtype=np.float32)
    relu7 = None
    dropout2 = None
    fc3_weight = np.zeros((num_classes, 4096), dtype=np.float32)
    fc3_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000):
    x = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    x = np.maximum(x, 0)
    x = _maxpool2d(x, maxpool1_kernel_size, maxpool1_stride, maxpool1_padding)
    x = _conv2d(x, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups)
    x = np.maximum(x, 0)
    x = _maxpool2d(x, maxpool2_kernel_size, maxpool2_stride, maxpool2_padding)
    x = _conv2d(x, conv3_weight, conv3_bias, conv3_stride, conv3_padding, conv3_dilation, conv3_groups)
    x = np.maximum(x, 0)
    x = _conv2d(x, conv4_weight, conv4_bias, conv4_stride, conv4_padding, conv4_dilation, conv4_groups)
    x = np.maximum(x, 0)
    x = _conv2d(x, conv5_weight, conv5_bias, conv5_stride, conv5_padding, conv5_dilation, conv5_groups)
    x = np.maximum(x, 0)
    x = _maxpool2d(x, maxpool3_kernel_size, maxpool3_stride, maxpool3_padding)
    x = np.reshape(x, (x.shape[0], -1))
    x = ((x) @ fc1_weight.T + fc1_bias)
    x = np.maximum(x, 0)
    x = x
    x = ((x) @ fc2_weight.T + fc2_bias)
    x = np.maximum(x, 0)
    x = x
    x = ((x) @ fc3_weight.T + fc3_bias)
    return x

