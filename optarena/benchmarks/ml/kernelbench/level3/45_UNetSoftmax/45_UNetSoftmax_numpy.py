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


def _conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(output_padding, int): output_padding = (output_padding, output_padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    n, c_in, h, w = x.shape
    _, c_out_per_group, kh, kw = weight.shape
    c_out = c_out_per_group * groups
    oh = (h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kh - 1) + output_padding[0] + 1
    ow = (w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kw - 1) + output_padding[1] + 1
    out = np.zeros((n, c_out, oh, ow), dtype=x.dtype)
    in_per_group = c_in // groups
    for b in range(n):
        for ic in range(c_in):
            g = ic // in_per_group
            for iy in range(h):
                for ix in range(w):
                    for ky in range(kh):
                        oy = iy * stride[0] - padding[0] + ky * dilation[0]
                        if 0 <= oy < oh:
                            for kx in range(kw):
                                ox = ix * stride[1] - padding[1] + kx * dilation[1]
                                if 0 <= ox < ow:
                                    for ocg in range(c_out_per_group):
                                        out[b, g * c_out_per_group + ocg, oy, ox] += x[b, ic, iy, ix] * weight[ic, ocg, ky, kx]
    out += bias.reshape(1, -1, 1, 1)
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


def _softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def _encoder1_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, encoder1_double_conv_0_weight, encoder1_double_conv_0_bias, encoder1_double_conv_0_stride, encoder1_double_conv_0_padding, encoder1_double_conv_0_dilation, encoder1_double_conv_0_groups), encoder1_double_conv_1_weight, encoder1_double_conv_1_bias, encoder1_double_conv_1_running_mean, encoder1_double_conv_1_running_var, encoder1_double_conv_1_eps), axis=encoder1_double_conv_2_dim), encoder1_double_conv_3_weight, encoder1_double_conv_3_bias, encoder1_double_conv_3_stride, encoder1_double_conv_3_padding, encoder1_double_conv_3_dilation, encoder1_double_conv_3_groups), encoder1_double_conv_4_weight, encoder1_double_conv_4_bias, encoder1_double_conv_4_running_mean, encoder1_double_conv_4_running_var, encoder1_double_conv_4_eps), axis=encoder1_double_conv_5_dim)

def _encoder2_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, encoder2_double_conv_0_weight, encoder2_double_conv_0_bias, encoder2_double_conv_0_stride, encoder2_double_conv_0_padding, encoder2_double_conv_0_dilation, encoder2_double_conv_0_groups), encoder2_double_conv_1_weight, encoder2_double_conv_1_bias, encoder2_double_conv_1_running_mean, encoder2_double_conv_1_running_var, encoder2_double_conv_1_eps), axis=encoder2_double_conv_2_dim), encoder2_double_conv_3_weight, encoder2_double_conv_3_bias, encoder2_double_conv_3_stride, encoder2_double_conv_3_padding, encoder2_double_conv_3_dilation, encoder2_double_conv_3_groups), encoder2_double_conv_4_weight, encoder2_double_conv_4_bias, encoder2_double_conv_4_running_mean, encoder2_double_conv_4_running_var, encoder2_double_conv_4_eps), axis=encoder2_double_conv_5_dim)

def _encoder3_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, encoder3_double_conv_0_weight, encoder3_double_conv_0_bias, encoder3_double_conv_0_stride, encoder3_double_conv_0_padding, encoder3_double_conv_0_dilation, encoder3_double_conv_0_groups), encoder3_double_conv_1_weight, encoder3_double_conv_1_bias, encoder3_double_conv_1_running_mean, encoder3_double_conv_1_running_var, encoder3_double_conv_1_eps), axis=encoder3_double_conv_2_dim), encoder3_double_conv_3_weight, encoder3_double_conv_3_bias, encoder3_double_conv_3_stride, encoder3_double_conv_3_padding, encoder3_double_conv_3_dilation, encoder3_double_conv_3_groups), encoder3_double_conv_4_weight, encoder3_double_conv_4_bias, encoder3_double_conv_4_running_mean, encoder3_double_conv_4_running_var, encoder3_double_conv_4_eps), axis=encoder3_double_conv_5_dim)

def _encoder4_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, encoder4_double_conv_0_weight, encoder4_double_conv_0_bias, encoder4_double_conv_0_stride, encoder4_double_conv_0_padding, encoder4_double_conv_0_dilation, encoder4_double_conv_0_groups), encoder4_double_conv_1_weight, encoder4_double_conv_1_bias, encoder4_double_conv_1_running_mean, encoder4_double_conv_1_running_var, encoder4_double_conv_1_eps), axis=encoder4_double_conv_2_dim), encoder4_double_conv_3_weight, encoder4_double_conv_3_bias, encoder4_double_conv_3_stride, encoder4_double_conv_3_padding, encoder4_double_conv_3_dilation, encoder4_double_conv_3_groups), encoder4_double_conv_4_weight, encoder4_double_conv_4_bias, encoder4_double_conv_4_running_mean, encoder4_double_conv_4_running_var, encoder4_double_conv_4_eps), axis=encoder4_double_conv_5_dim)

def _bottleneck_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, bottleneck_double_conv_0_weight, bottleneck_double_conv_0_bias, bottleneck_double_conv_0_stride, bottleneck_double_conv_0_padding, bottleneck_double_conv_0_dilation, bottleneck_double_conv_0_groups), bottleneck_double_conv_1_weight, bottleneck_double_conv_1_bias, bottleneck_double_conv_1_running_mean, bottleneck_double_conv_1_running_var, bottleneck_double_conv_1_eps), axis=bottleneck_double_conv_2_dim), bottleneck_double_conv_3_weight, bottleneck_double_conv_3_bias, bottleneck_double_conv_3_stride, bottleneck_double_conv_3_padding, bottleneck_double_conv_3_dilation, bottleneck_double_conv_3_groups), bottleneck_double_conv_4_weight, bottleneck_double_conv_4_bias, bottleneck_double_conv_4_running_mean, bottleneck_double_conv_4_running_var, bottleneck_double_conv_4_eps), axis=bottleneck_double_conv_5_dim)

def _decoder4_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, decoder4_double_conv_0_weight, decoder4_double_conv_0_bias, decoder4_double_conv_0_stride, decoder4_double_conv_0_padding, decoder4_double_conv_0_dilation, decoder4_double_conv_0_groups), decoder4_double_conv_1_weight, decoder4_double_conv_1_bias, decoder4_double_conv_1_running_mean, decoder4_double_conv_1_running_var, decoder4_double_conv_1_eps), axis=decoder4_double_conv_2_dim), decoder4_double_conv_3_weight, decoder4_double_conv_3_bias, decoder4_double_conv_3_stride, decoder4_double_conv_3_padding, decoder4_double_conv_3_dilation, decoder4_double_conv_3_groups), decoder4_double_conv_4_weight, decoder4_double_conv_4_bias, decoder4_double_conv_4_running_mean, decoder4_double_conv_4_running_var, decoder4_double_conv_4_eps), axis=decoder4_double_conv_5_dim)

def _decoder3_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, decoder3_double_conv_0_weight, decoder3_double_conv_0_bias, decoder3_double_conv_0_stride, decoder3_double_conv_0_padding, decoder3_double_conv_0_dilation, decoder3_double_conv_0_groups), decoder3_double_conv_1_weight, decoder3_double_conv_1_bias, decoder3_double_conv_1_running_mean, decoder3_double_conv_1_running_var, decoder3_double_conv_1_eps), axis=decoder3_double_conv_2_dim), decoder3_double_conv_3_weight, decoder3_double_conv_3_bias, decoder3_double_conv_3_stride, decoder3_double_conv_3_padding, decoder3_double_conv_3_dilation, decoder3_double_conv_3_groups), decoder3_double_conv_4_weight, decoder3_double_conv_4_bias, decoder3_double_conv_4_running_mean, decoder3_double_conv_4_running_var, decoder3_double_conv_4_eps), axis=decoder3_double_conv_5_dim)

def _decoder2_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, decoder2_double_conv_0_weight, decoder2_double_conv_0_bias, decoder2_double_conv_0_stride, decoder2_double_conv_0_padding, decoder2_double_conv_0_dilation, decoder2_double_conv_0_groups), decoder2_double_conv_1_weight, decoder2_double_conv_1_bias, decoder2_double_conv_1_running_mean, decoder2_double_conv_1_running_var, decoder2_double_conv_1_eps), axis=decoder2_double_conv_2_dim), decoder2_double_conv_3_weight, decoder2_double_conv_3_bias, decoder2_double_conv_3_stride, decoder2_double_conv_3_padding, decoder2_double_conv_3_dilation, decoder2_double_conv_3_groups), decoder2_double_conv_4_weight, decoder2_double_conv_4_bias, decoder2_double_conv_4_running_mean, decoder2_double_conv_4_running_var, decoder2_double_conv_4_eps), axis=decoder2_double_conv_5_dim)

def _decoder1_forward(x):
    return _softmax(_batch_norm(_conv2d(_softmax(_batch_norm(_conv2d(x, decoder1_double_conv_0_weight, decoder1_double_conv_0_bias, decoder1_double_conv_0_stride, decoder1_double_conv_0_padding, decoder1_double_conv_0_dilation, decoder1_double_conv_0_groups), decoder1_double_conv_1_weight, decoder1_double_conv_1_bias, decoder1_double_conv_1_running_mean, decoder1_double_conv_1_running_var, decoder1_double_conv_1_eps), axis=decoder1_double_conv_2_dim), decoder1_double_conv_3_weight, decoder1_double_conv_3_bias, decoder1_double_conv_3_stride, decoder1_double_conv_3_padding, decoder1_double_conv_3_dilation, decoder1_double_conv_3_groups), decoder1_double_conv_4_weight, decoder1_double_conv_4_bias, decoder1_double_conv_4_running_mean, decoder1_double_conv_4_running_var, decoder1_double_conv_4_eps), axis=decoder1_double_conv_5_dim)

def init(in_channels, out_channels, features):
    global encoder1_double_conv_0_weight, encoder1_double_conv_0_bias, encoder1_double_conv_0_stride, encoder1_double_conv_0_padding, encoder1_double_conv_0_dilation, encoder1_double_conv_0_groups, encoder1_double_conv_1_weight, encoder1_double_conv_1_bias, encoder1_double_conv_1_running_mean, encoder1_double_conv_1_running_var, encoder1_double_conv_1_eps, encoder1_double_conv_2_dim, encoder1_double_conv_3_weight, encoder1_double_conv_3_bias, encoder1_double_conv_3_stride, encoder1_double_conv_3_padding, encoder1_double_conv_3_dilation, encoder1_double_conv_3_groups, encoder1_double_conv_4_weight, encoder1_double_conv_4_bias, encoder1_double_conv_4_running_mean, encoder1_double_conv_4_running_var, encoder1_double_conv_4_eps, encoder1_double_conv_5_dim, pool1_kernel_size, pool1_stride, pool1_padding, encoder2_double_conv_0_weight, encoder2_double_conv_0_bias, encoder2_double_conv_0_stride, encoder2_double_conv_0_padding, encoder2_double_conv_0_dilation, encoder2_double_conv_0_groups, encoder2_double_conv_1_weight, encoder2_double_conv_1_bias, encoder2_double_conv_1_running_mean, encoder2_double_conv_1_running_var, encoder2_double_conv_1_eps, encoder2_double_conv_2_dim, encoder2_double_conv_3_weight, encoder2_double_conv_3_bias, encoder2_double_conv_3_stride, encoder2_double_conv_3_padding, encoder2_double_conv_3_dilation, encoder2_double_conv_3_groups, encoder2_double_conv_4_weight, encoder2_double_conv_4_bias, encoder2_double_conv_4_running_mean, encoder2_double_conv_4_running_var, encoder2_double_conv_4_eps, encoder2_double_conv_5_dim, pool2_kernel_size, pool2_stride, pool2_padding, encoder3_double_conv_0_weight, encoder3_double_conv_0_bias, encoder3_double_conv_0_stride, encoder3_double_conv_0_padding, encoder3_double_conv_0_dilation, encoder3_double_conv_0_groups, encoder3_double_conv_1_weight, encoder3_double_conv_1_bias, encoder3_double_conv_1_running_mean, encoder3_double_conv_1_running_var, encoder3_double_conv_1_eps, encoder3_double_conv_2_dim, encoder3_double_conv_3_weight, encoder3_double_conv_3_bias, encoder3_double_conv_3_stride, encoder3_double_conv_3_padding, encoder3_double_conv_3_dilation, encoder3_double_conv_3_groups, encoder3_double_conv_4_weight, encoder3_double_conv_4_bias, encoder3_double_conv_4_running_mean, encoder3_double_conv_4_running_var, encoder3_double_conv_4_eps, encoder3_double_conv_5_dim, pool3_kernel_size, pool3_stride, pool3_padding, encoder4_double_conv_0_weight, encoder4_double_conv_0_bias, encoder4_double_conv_0_stride, encoder4_double_conv_0_padding, encoder4_double_conv_0_dilation, encoder4_double_conv_0_groups, encoder4_double_conv_1_weight, encoder4_double_conv_1_bias, encoder4_double_conv_1_running_mean, encoder4_double_conv_1_running_var, encoder4_double_conv_1_eps, encoder4_double_conv_2_dim, encoder4_double_conv_3_weight, encoder4_double_conv_3_bias, encoder4_double_conv_3_stride, encoder4_double_conv_3_padding, encoder4_double_conv_3_dilation, encoder4_double_conv_3_groups, encoder4_double_conv_4_weight, encoder4_double_conv_4_bias, encoder4_double_conv_4_running_mean, encoder4_double_conv_4_running_var, encoder4_double_conv_4_eps, encoder4_double_conv_5_dim, pool4_kernel_size, pool4_stride, pool4_padding, bottleneck_double_conv_0_weight, bottleneck_double_conv_0_bias, bottleneck_double_conv_0_stride, bottleneck_double_conv_0_padding, bottleneck_double_conv_0_dilation, bottleneck_double_conv_0_groups, bottleneck_double_conv_1_weight, bottleneck_double_conv_1_bias, bottleneck_double_conv_1_running_mean, bottleneck_double_conv_1_running_var, bottleneck_double_conv_1_eps, bottleneck_double_conv_2_dim, bottleneck_double_conv_3_weight, bottleneck_double_conv_3_bias, bottleneck_double_conv_3_stride, bottleneck_double_conv_3_padding, bottleneck_double_conv_3_dilation, bottleneck_double_conv_3_groups, bottleneck_double_conv_4_weight, bottleneck_double_conv_4_bias, bottleneck_double_conv_4_running_mean, bottleneck_double_conv_4_running_var, bottleneck_double_conv_4_eps, bottleneck_double_conv_5_dim, upconv4_weight, upconv4_bias, upconv4_stride, upconv4_padding, upconv4_dilation, upconv4_groups, upconv4_output_padding, decoder4_double_conv_0_weight, decoder4_double_conv_0_bias, decoder4_double_conv_0_stride, decoder4_double_conv_0_padding, decoder4_double_conv_0_dilation, decoder4_double_conv_0_groups, decoder4_double_conv_1_weight, decoder4_double_conv_1_bias, decoder4_double_conv_1_running_mean, decoder4_double_conv_1_running_var, decoder4_double_conv_1_eps, decoder4_double_conv_2_dim, decoder4_double_conv_3_weight, decoder4_double_conv_3_bias, decoder4_double_conv_3_stride, decoder4_double_conv_3_padding, decoder4_double_conv_3_dilation, decoder4_double_conv_3_groups, decoder4_double_conv_4_weight, decoder4_double_conv_4_bias, decoder4_double_conv_4_running_mean, decoder4_double_conv_4_running_var, decoder4_double_conv_4_eps, decoder4_double_conv_5_dim, upconv3_weight, upconv3_bias, upconv3_stride, upconv3_padding, upconv3_dilation, upconv3_groups, upconv3_output_padding, decoder3_double_conv_0_weight, decoder3_double_conv_0_bias, decoder3_double_conv_0_stride, decoder3_double_conv_0_padding, decoder3_double_conv_0_dilation, decoder3_double_conv_0_groups, decoder3_double_conv_1_weight, decoder3_double_conv_1_bias, decoder3_double_conv_1_running_mean, decoder3_double_conv_1_running_var, decoder3_double_conv_1_eps, decoder3_double_conv_2_dim, decoder3_double_conv_3_weight, decoder3_double_conv_3_bias, decoder3_double_conv_3_stride, decoder3_double_conv_3_padding, decoder3_double_conv_3_dilation, decoder3_double_conv_3_groups, decoder3_double_conv_4_weight, decoder3_double_conv_4_bias, decoder3_double_conv_4_running_mean, decoder3_double_conv_4_running_var, decoder3_double_conv_4_eps, decoder3_double_conv_5_dim, upconv2_weight, upconv2_bias, upconv2_stride, upconv2_padding, upconv2_dilation, upconv2_groups, upconv2_output_padding, decoder2_double_conv_0_weight, decoder2_double_conv_0_bias, decoder2_double_conv_0_stride, decoder2_double_conv_0_padding, decoder2_double_conv_0_dilation, decoder2_double_conv_0_groups, decoder2_double_conv_1_weight, decoder2_double_conv_1_bias, decoder2_double_conv_1_running_mean, decoder2_double_conv_1_running_var, decoder2_double_conv_1_eps, decoder2_double_conv_2_dim, decoder2_double_conv_3_weight, decoder2_double_conv_3_bias, decoder2_double_conv_3_stride, decoder2_double_conv_3_padding, decoder2_double_conv_3_dilation, decoder2_double_conv_3_groups, decoder2_double_conv_4_weight, decoder2_double_conv_4_bias, decoder2_double_conv_4_running_mean, decoder2_double_conv_4_running_var, decoder2_double_conv_4_eps, decoder2_double_conv_5_dim, upconv1_weight, upconv1_bias, upconv1_stride, upconv1_padding, upconv1_dilation, upconv1_groups, upconv1_output_padding, decoder1_double_conv_0_weight, decoder1_double_conv_0_bias, decoder1_double_conv_0_stride, decoder1_double_conv_0_padding, decoder1_double_conv_0_dilation, decoder1_double_conv_0_groups, decoder1_double_conv_1_weight, decoder1_double_conv_1_bias, decoder1_double_conv_1_running_mean, decoder1_double_conv_1_running_var, decoder1_double_conv_1_eps, decoder1_double_conv_2_dim, decoder1_double_conv_3_weight, decoder1_double_conv_3_bias, decoder1_double_conv_3_stride, decoder1_double_conv_3_padding, decoder1_double_conv_3_dilation, decoder1_double_conv_3_groups, decoder1_double_conv_4_weight, decoder1_double_conv_4_bias, decoder1_double_conv_4_running_mean, decoder1_double_conv_4_running_var, decoder1_double_conv_4_eps, decoder1_double_conv_5_dim, final_conv_weight, final_conv_bias, final_conv_stride, final_conv_padding, final_conv_dilation, final_conv_groups
    encoder1_double_conv_0_weight = np.zeros((features, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder1_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    encoder1_double_conv_0_stride = 1
    encoder1_double_conv_0_padding = 1
    encoder1_double_conv_0_dilation = 1
    encoder1_double_conv_0_groups = 1
    encoder1_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    encoder1_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    encoder1_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    encoder1_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    encoder1_double_conv_1_eps = 1e-5
    encoder1_double_conv_2_dim = -1
    encoder1_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder1_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    encoder1_double_conv_3_stride = 1
    encoder1_double_conv_3_padding = 1
    encoder1_double_conv_3_dilation = 1
    encoder1_double_conv_3_groups = 1
    encoder1_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    encoder1_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    encoder1_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    encoder1_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    encoder1_double_conv_4_eps = 1e-5
    encoder1_double_conv_5_dim = -1
    pool1_kernel_size = 2
    pool1_stride = 2
    pool1_padding = 0
    encoder2_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder2_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    encoder2_double_conv_0_stride = 1
    encoder2_double_conv_0_padding = 1
    encoder2_double_conv_0_dilation = 1
    encoder2_double_conv_0_groups = 1
    encoder2_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    encoder2_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    encoder2_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    encoder2_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    encoder2_double_conv_1_eps = 1e-5
    encoder2_double_conv_2_dim = -1
    encoder2_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder2_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    encoder2_double_conv_3_stride = 1
    encoder2_double_conv_3_padding = 1
    encoder2_double_conv_3_dilation = 1
    encoder2_double_conv_3_groups = 1
    encoder2_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    encoder2_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    encoder2_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    encoder2_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    encoder2_double_conv_4_eps = 1e-5
    encoder2_double_conv_5_dim = -1
    pool2_kernel_size = 2
    pool2_stride = 2
    pool2_padding = 0
    encoder3_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder3_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    encoder3_double_conv_0_stride = 1
    encoder3_double_conv_0_padding = 1
    encoder3_double_conv_0_dilation = 1
    encoder3_double_conv_0_groups = 1
    encoder3_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    encoder3_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    encoder3_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    encoder3_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    encoder3_double_conv_1_eps = 1e-5
    encoder3_double_conv_2_dim = -1
    encoder3_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder3_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    encoder3_double_conv_3_stride = 1
    encoder3_double_conv_3_padding = 1
    encoder3_double_conv_3_dilation = 1
    encoder3_double_conv_3_groups = 1
    encoder3_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    encoder3_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    encoder3_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    encoder3_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    encoder3_double_conv_4_eps = 1e-5
    encoder3_double_conv_5_dim = -1
    pool3_kernel_size = 2
    pool3_stride = 2
    pool3_padding = 0
    encoder4_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder4_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    encoder4_double_conv_0_stride = 1
    encoder4_double_conv_0_padding = 1
    encoder4_double_conv_0_dilation = 1
    encoder4_double_conv_0_groups = 1
    encoder4_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    encoder4_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    encoder4_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    encoder4_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    encoder4_double_conv_1_eps = 1e-5
    encoder4_double_conv_2_dim = -1
    encoder4_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    encoder4_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    encoder4_double_conv_3_stride = 1
    encoder4_double_conv_3_padding = 1
    encoder4_double_conv_3_dilation = 1
    encoder4_double_conv_3_groups = 1
    encoder4_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    encoder4_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    encoder4_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    encoder4_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    encoder4_double_conv_4_eps = 1e-5
    encoder4_double_conv_5_dim = -1
    pool4_kernel_size = 2
    pool4_stride = 2
    pool4_padding = 0
    bottleneck_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    bottleneck_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    bottleneck_double_conv_0_stride = 1
    bottleneck_double_conv_0_padding = 1
    bottleneck_double_conv_0_dilation = 1
    bottleneck_double_conv_0_groups = 1
    bottleneck_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    bottleneck_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    bottleneck_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    bottleneck_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    bottleneck_double_conv_1_eps = 1e-5
    bottleneck_double_conv_2_dim = -1
    bottleneck_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    bottleneck_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    bottleneck_double_conv_3_stride = 1
    bottleneck_double_conv_3_padding = 1
    bottleneck_double_conv_3_dilation = 1
    bottleneck_double_conv_3_groups = 1
    bottleneck_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    bottleneck_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    bottleneck_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    bottleneck_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    bottleneck_double_conv_4_eps = 1e-5
    bottleneck_double_conv_5_dim = -1
    upconv4_weight = np.zeros((features * 16, features * 8 // 1) + _as_tuple(2, 2), dtype=np.float32)
    upconv4_bias = np.zeros((features * 8,), dtype=np.float32)
    upconv4_stride = 2
    upconv4_padding = 0
    upconv4_dilation = 1
    upconv4_groups = 1
    upconv4_output_padding = 0
    decoder4_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder4_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    decoder4_double_conv_0_stride = 1
    decoder4_double_conv_0_padding = 1
    decoder4_double_conv_0_dilation = 1
    decoder4_double_conv_0_groups = 1
    decoder4_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    decoder4_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    decoder4_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    decoder4_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    decoder4_double_conv_1_eps = 1e-5
    decoder4_double_conv_2_dim = -1
    decoder4_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder4_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    decoder4_double_conv_3_stride = 1
    decoder4_double_conv_3_padding = 1
    decoder4_double_conv_3_dilation = 1
    decoder4_double_conv_3_groups = 1
    decoder4_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    decoder4_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    decoder4_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    decoder4_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    decoder4_double_conv_4_eps = 1e-5
    decoder4_double_conv_5_dim = -1
    upconv3_weight = np.zeros((features * 8, features * 4 // 1) + _as_tuple(2, 2), dtype=np.float32)
    upconv3_bias = np.zeros((features * 4,), dtype=np.float32)
    upconv3_stride = 2
    upconv3_padding = 0
    upconv3_dilation = 1
    upconv3_groups = 1
    upconv3_output_padding = 0
    decoder3_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder3_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    decoder3_double_conv_0_stride = 1
    decoder3_double_conv_0_padding = 1
    decoder3_double_conv_0_dilation = 1
    decoder3_double_conv_0_groups = 1
    decoder3_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    decoder3_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    decoder3_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    decoder3_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    decoder3_double_conv_1_eps = 1e-5
    decoder3_double_conv_2_dim = -1
    decoder3_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder3_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    decoder3_double_conv_3_stride = 1
    decoder3_double_conv_3_padding = 1
    decoder3_double_conv_3_dilation = 1
    decoder3_double_conv_3_groups = 1
    decoder3_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    decoder3_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    decoder3_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    decoder3_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    decoder3_double_conv_4_eps = 1e-5
    decoder3_double_conv_5_dim = -1
    upconv2_weight = np.zeros((features * 4, features * 2 // 1) + _as_tuple(2, 2), dtype=np.float32)
    upconv2_bias = np.zeros((features * 2,), dtype=np.float32)
    upconv2_stride = 2
    upconv2_padding = 0
    upconv2_dilation = 1
    upconv2_groups = 1
    upconv2_output_padding = 0
    decoder2_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder2_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    decoder2_double_conv_0_stride = 1
    decoder2_double_conv_0_padding = 1
    decoder2_double_conv_0_dilation = 1
    decoder2_double_conv_0_groups = 1
    decoder2_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    decoder2_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    decoder2_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    decoder2_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    decoder2_double_conv_1_eps = 1e-5
    decoder2_double_conv_2_dim = -1
    decoder2_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder2_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    decoder2_double_conv_3_stride = 1
    decoder2_double_conv_3_padding = 1
    decoder2_double_conv_3_dilation = 1
    decoder2_double_conv_3_groups = 1
    decoder2_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    decoder2_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    decoder2_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    decoder2_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    decoder2_double_conv_4_eps = 1e-5
    decoder2_double_conv_5_dim = -1
    upconv1_weight = np.zeros((features * 2, features // 1) + _as_tuple(2, 2), dtype=np.float32)
    upconv1_bias = np.zeros((features,), dtype=np.float32)
    upconv1_stride = 2
    upconv1_padding = 0
    upconv1_dilation = 1
    upconv1_groups = 1
    upconv1_output_padding = 0
    decoder1_double_conv_0_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder1_double_conv_0_bias = np.zeros((features,), dtype=np.float32)
    decoder1_double_conv_0_stride = 1
    decoder1_double_conv_0_padding = 1
    decoder1_double_conv_0_dilation = 1
    decoder1_double_conv_0_groups = 1
    decoder1_double_conv_1_weight = np.ones((features,), dtype=np.float32)
    decoder1_double_conv_1_bias = np.zeros((features,), dtype=np.float32)
    decoder1_double_conv_1_running_mean = np.zeros((features,), dtype=np.float32)
    decoder1_double_conv_1_running_var = np.ones((features,), dtype=np.float32)
    decoder1_double_conv_1_eps = 1e-5
    decoder1_double_conv_2_dim = -1
    decoder1_double_conv_3_weight = np.zeros((features, features // 1) + _as_tuple(3, 2), dtype=np.float32)
    decoder1_double_conv_3_bias = np.zeros((features,), dtype=np.float32)
    decoder1_double_conv_3_stride = 1
    decoder1_double_conv_3_padding = 1
    decoder1_double_conv_3_dilation = 1
    decoder1_double_conv_3_groups = 1
    decoder1_double_conv_4_weight = np.ones((features,), dtype=np.float32)
    decoder1_double_conv_4_bias = np.zeros((features,), dtype=np.float32)
    decoder1_double_conv_4_running_mean = np.zeros((features,), dtype=np.float32)
    decoder1_double_conv_4_running_var = np.ones((features,), dtype=np.float32)
    decoder1_double_conv_4_eps = 1e-5
    decoder1_double_conv_5_dim = -1
    final_conv_weight = np.zeros((out_channels, features // 1) + _as_tuple(1, 2), dtype=np.float32)
    final_conv_bias = np.zeros((out_channels,), dtype=np.float32)
    final_conv_stride = 1
    final_conv_padding = 0
    final_conv_dilation = 1
    final_conv_groups = 1

def forward(x, in_channels, out_channels, features):
    enc1 = _encoder1_forward(x)
    enc2 = _encoder2_forward(_maxpool2d(enc1, pool1_kernel_size, pool1_stride, pool1_padding))
    enc3 = _encoder3_forward(_maxpool2d(enc2, pool2_kernel_size, pool2_stride, pool2_padding))
    enc4 = _encoder4_forward(_maxpool2d(enc3, pool3_kernel_size, pool3_stride, pool3_padding))
    bottleneck = _bottleneck_forward(_maxpool2d(enc4, pool4_kernel_size, pool4_stride, pool4_padding))
    dec4 = _conv_transpose2d(bottleneck, upconv4_weight, upconv4_bias, upconv4_stride, upconv4_padding, upconv4_output_padding, upconv4_dilation, upconv4_groups)
    dec4 = np.concatenate((dec4, enc4), axis=1)
    dec4 = _decoder4_forward(dec4)
    dec3 = _conv_transpose2d(dec4, upconv3_weight, upconv3_bias, upconv3_stride, upconv3_padding, upconv3_output_padding, upconv3_dilation, upconv3_groups)
    dec3 = np.concatenate((dec3, enc3), axis=1)
    dec3 = _decoder3_forward(dec3)
    dec2 = _conv_transpose2d(dec3, upconv2_weight, upconv2_bias, upconv2_stride, upconv2_padding, upconv2_output_padding, upconv2_dilation, upconv2_groups)
    dec2 = np.concatenate((dec2, enc2), axis=1)
    dec2 = _decoder2_forward(dec2)
    dec1 = _conv_transpose2d(dec2, upconv1_weight, upconv1_bias, upconv1_stride, upconv1_padding, upconv1_output_padding, upconv1_dilation, upconv1_groups)
    dec1 = np.concatenate((dec1, enc1), axis=1)
    dec1 = _decoder1_forward(dec1)
    return _conv2d(dec1, final_conv_weight, final_conv_bias, final_conv_stride, final_conv_padding, final_conv_dilation, final_conv_groups)

