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

def _layer1_0_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer1_0_conv2_weight, layer1_0_conv2_bias, layer1_0_conv2_stride, layer1_0_conv2_padding, layer1_0_conv2_dilation, layer1_0_conv2_groups)
    out = _batch_norm(out, layer1_0_bn2_weight, layer1_0_bn2_bias, layer1_0_bn2_running_mean, layer1_0_bn2_running_var, layer1_0_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def _layer1_1_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer1_1_conv2_weight, layer1_1_conv2_bias, layer1_1_conv2_stride, layer1_1_conv2_padding, layer1_1_conv2_dilation, layer1_1_conv2_groups)
    out = _batch_norm(out, layer1_1_bn2_weight, layer1_1_bn2_bias, layer1_1_bn2_running_mean, layer1_1_bn2_running_var, layer1_1_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def _layer2_0_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer2_0_conv2_weight, layer2_0_conv2_bias, layer2_0_conv2_stride, layer2_0_conv2_padding, layer2_0_conv2_dilation, layer2_0_conv2_groups)
    out = _batch_norm(out, layer2_0_bn2_weight, layer2_0_bn2_bias, layer2_0_bn2_running_mean, layer2_0_bn2_running_var, layer2_0_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def _layer2_1_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer2_1_conv2_weight, layer2_1_conv2_bias, layer2_1_conv2_stride, layer2_1_conv2_padding, layer2_1_conv2_dilation, layer2_1_conv2_groups)
    out = _batch_norm(out, layer2_1_bn2_weight, layer2_1_bn2_bias, layer2_1_bn2_running_mean, layer2_1_bn2_running_var, layer2_1_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def _layer3_0_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer3_0_conv2_weight, layer3_0_conv2_bias, layer3_0_conv2_stride, layer3_0_conv2_padding, layer3_0_conv2_dilation, layer3_0_conv2_groups)
    out = _batch_norm(out, layer3_0_bn2_weight, layer3_0_bn2_bias, layer3_0_bn2_running_mean, layer3_0_bn2_running_var, layer3_0_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def _layer3_1_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer3_1_conv2_weight, layer3_1_conv2_bias, layer3_1_conv2_stride, layer3_1_conv2_padding, layer3_1_conv2_dilation, layer3_1_conv2_groups)
    out = _batch_norm(out, layer3_1_bn2_weight, layer3_1_bn2_bias, layer3_1_bn2_running_mean, layer3_1_bn2_running_var, layer3_1_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def _layer4_0_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer4_0_conv2_weight, layer4_0_conv2_bias, layer4_0_conv2_stride, layer4_0_conv2_padding, layer4_0_conv2_dilation, layer4_0_conv2_groups)
    out = _batch_norm(out, layer4_0_bn2_weight, layer4_0_bn2_bias, layer4_0_bn2_running_mean, layer4_0_bn2_running_var, layer4_0_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def _layer4_1_forward(x):
    identity = x
    out = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    out = _batch_norm(out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    out = np.maximum(out, 0)
    out = _conv2d(out, layer4_1_conv2_weight, layer4_1_conv2_bias, layer4_1_conv2_stride, layer4_1_conv2_padding, layer4_1_conv2_dilation, layer4_1_conv2_groups)
    out = _batch_norm(out, layer4_1_bn2_weight, layer4_1_bn2_bias, layer4_1_bn2_running_mean, layer4_1_bn2_running_var, layer4_1_bn2_eps)
    out += identity
    out = np.maximum(out, 0)
    return out

def init(num_classes=1000):
    global in_channels, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps, relu, maxpool_kernel_size, maxpool_stride, maxpool_padding, layer1_0_conv1_weight, layer1_0_conv1_bias, layer1_0_conv1_stride, layer1_0_conv1_padding, layer1_0_conv1_dilation, layer1_0_conv1_groups, layer1_0_bn1_weight, layer1_0_bn1_bias, layer1_0_bn1_running_mean, layer1_0_bn1_running_var, layer1_0_bn1_eps, layer1_0_relu, layer1_0_conv2_weight, layer1_0_conv2_bias, layer1_0_conv2_stride, layer1_0_conv2_padding, layer1_0_conv2_dilation, layer1_0_conv2_groups, layer1_0_bn2_weight, layer1_0_bn2_bias, layer1_0_bn2_running_mean, layer1_0_bn2_running_var, layer1_0_bn2_eps, layer1_0_downsample, layer1_0_stride, layer1_1_conv1_weight, layer1_1_conv1_bias, layer1_1_conv1_stride, layer1_1_conv1_padding, layer1_1_conv1_dilation, layer1_1_conv1_groups, layer1_1_bn1_weight, layer1_1_bn1_bias, layer1_1_bn1_running_mean, layer1_1_bn1_running_var, layer1_1_bn1_eps, layer1_1_relu, layer1_1_conv2_weight, layer1_1_conv2_bias, layer1_1_conv2_stride, layer1_1_conv2_padding, layer1_1_conv2_dilation, layer1_1_conv2_groups, layer1_1_bn2_weight, layer1_1_bn2_bias, layer1_1_bn2_running_mean, layer1_1_bn2_running_var, layer1_1_bn2_eps, layer1_1_downsample, layer1_1_stride, layer2_0_conv1_weight, layer2_0_conv1_bias, layer2_0_conv1_stride, layer2_0_conv1_padding, layer2_0_conv1_dilation, layer2_0_conv1_groups, layer2_0_bn1_weight, layer2_0_bn1_bias, layer2_0_bn1_running_mean, layer2_0_bn1_running_var, layer2_0_bn1_eps, layer2_0_relu, layer2_0_conv2_weight, layer2_0_conv2_bias, layer2_0_conv2_stride, layer2_0_conv2_padding, layer2_0_conv2_dilation, layer2_0_conv2_groups, layer2_0_bn2_weight, layer2_0_bn2_bias, layer2_0_bn2_running_mean, layer2_0_bn2_running_var, layer2_0_bn2_eps, layer2_0_downsample, layer2_0_stride, layer2_1_conv1_weight, layer2_1_conv1_bias, layer2_1_conv1_stride, layer2_1_conv1_padding, layer2_1_conv1_dilation, layer2_1_conv1_groups, layer2_1_bn1_weight, layer2_1_bn1_bias, layer2_1_bn1_running_mean, layer2_1_bn1_running_var, layer2_1_bn1_eps, layer2_1_relu, layer2_1_conv2_weight, layer2_1_conv2_bias, layer2_1_conv2_stride, layer2_1_conv2_padding, layer2_1_conv2_dilation, layer2_1_conv2_groups, layer2_1_bn2_weight, layer2_1_bn2_bias, layer2_1_bn2_running_mean, layer2_1_bn2_running_var, layer2_1_bn2_eps, layer2_1_downsample, layer2_1_stride, layer3_0_conv1_weight, layer3_0_conv1_bias, layer3_0_conv1_stride, layer3_0_conv1_padding, layer3_0_conv1_dilation, layer3_0_conv1_groups, layer3_0_bn1_weight, layer3_0_bn1_bias, layer3_0_bn1_running_mean, layer3_0_bn1_running_var, layer3_0_bn1_eps, layer3_0_relu, layer3_0_conv2_weight, layer3_0_conv2_bias, layer3_0_conv2_stride, layer3_0_conv2_padding, layer3_0_conv2_dilation, layer3_0_conv2_groups, layer3_0_bn2_weight, layer3_0_bn2_bias, layer3_0_bn2_running_mean, layer3_0_bn2_running_var, layer3_0_bn2_eps, layer3_0_downsample, layer3_0_stride, layer3_1_conv1_weight, layer3_1_conv1_bias, layer3_1_conv1_stride, layer3_1_conv1_padding, layer3_1_conv1_dilation, layer3_1_conv1_groups, layer3_1_bn1_weight, layer3_1_bn1_bias, layer3_1_bn1_running_mean, layer3_1_bn1_running_var, layer3_1_bn1_eps, layer3_1_relu, layer3_1_conv2_weight, layer3_1_conv2_bias, layer3_1_conv2_stride, layer3_1_conv2_padding, layer3_1_conv2_dilation, layer3_1_conv2_groups, layer3_1_bn2_weight, layer3_1_bn2_bias, layer3_1_bn2_running_mean, layer3_1_bn2_running_var, layer3_1_bn2_eps, layer3_1_downsample, layer3_1_stride, layer4_0_conv1_weight, layer4_0_conv1_bias, layer4_0_conv1_stride, layer4_0_conv1_padding, layer4_0_conv1_dilation, layer4_0_conv1_groups, layer4_0_bn1_weight, layer4_0_bn1_bias, layer4_0_bn1_running_mean, layer4_0_bn1_running_var, layer4_0_bn1_eps, layer4_0_relu, layer4_0_conv2_weight, layer4_0_conv2_bias, layer4_0_conv2_stride, layer4_0_conv2_padding, layer4_0_conv2_dilation, layer4_0_conv2_groups, layer4_0_bn2_weight, layer4_0_bn2_bias, layer4_0_bn2_running_mean, layer4_0_bn2_running_var, layer4_0_bn2_eps, layer4_0_downsample, layer4_0_stride, layer4_1_conv1_weight, layer4_1_conv1_bias, layer4_1_conv1_stride, layer4_1_conv1_padding, layer4_1_conv1_dilation, layer4_1_conv1_groups, layer4_1_bn1_weight, layer4_1_bn1_bias, layer4_1_bn1_running_mean, layer4_1_bn1_running_var, layer4_1_bn1_eps, layer4_1_relu, layer4_1_conv2_weight, layer4_1_conv2_bias, layer4_1_conv2_stride, layer4_1_conv2_padding, layer4_1_conv2_dilation, layer4_1_conv2_groups, layer4_1_bn2_weight, layer4_1_bn2_bias, layer4_1_bn2_running_mean, layer4_1_bn2_running_var, layer4_1_bn2_eps, layer4_1_downsample, layer4_1_stride, avgpool_output_size, fc_weight, fc_bias
    in_channels = 64
    conv1_weight = np.zeros((64, 3 // 1) + _as_tuple(7, 2), dtype=np.float32)
    conv1_bias = np.zeros((64,), dtype=np.float32)
    conv1_stride = 2
    conv1_padding = 3
    conv1_dilation = 1
    conv1_groups = 1
    bn1_weight = np.ones((64,), dtype=np.float32)
    bn1_bias = np.zeros((64,), dtype=np.float32)
    bn1_running_mean = np.zeros((64,), dtype=np.float32)
    bn1_running_var = np.ones((64,), dtype=np.float32)
    bn1_eps = 1e-5
    relu = None
    maxpool_kernel_size = 3
    maxpool_stride = 2
    maxpool_padding = 1
    layer1_0_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer1_0_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer1_0_conv1_stride = 1
    layer1_0_conv1_padding = 1
    layer1_0_conv1_dilation = 1
    layer1_0_conv1_groups = 1
    layer1_0_bn1_weight = np.ones((64,), dtype=np.float32)
    layer1_0_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer1_0_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer1_0_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer1_0_bn1_eps = 1e-5
    layer1_0_relu = None
    layer1_0_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer1_0_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer1_0_conv2_stride = 1
    layer1_0_conv2_padding = 1
    layer1_0_conv2_dilation = 1
    layer1_0_conv2_groups = 1
    layer1_0_bn2_weight = np.ones((64,), dtype=np.float32)
    layer1_0_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer1_0_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer1_0_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer1_0_bn2_eps = 1e-5
    layer1_0_downsample = None
    layer1_0_stride = 1
    layer1_1_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer1_1_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer1_1_conv1_stride = 1
    layer1_1_conv1_padding = 1
    layer1_1_conv1_dilation = 1
    layer1_1_conv1_groups = 1
    layer1_1_bn1_weight = np.ones((64,), dtype=np.float32)
    layer1_1_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer1_1_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer1_1_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer1_1_bn1_eps = 1e-5
    layer1_1_relu = None
    layer1_1_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer1_1_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer1_1_conv2_stride = 1
    layer1_1_conv2_padding = 1
    layer1_1_conv2_dilation = 1
    layer1_1_conv2_groups = 1
    layer1_1_bn2_weight = np.ones((64,), dtype=np.float32)
    layer1_1_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer1_1_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer1_1_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer1_1_bn2_eps = 1e-5
    layer1_1_downsample = None
    layer1_1_stride = 1
    layer2_0_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer2_0_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer2_0_conv1_stride = 1
    layer2_0_conv1_padding = 1
    layer2_0_conv1_dilation = 1
    layer2_0_conv1_groups = 1
    layer2_0_bn1_weight = np.ones((64,), dtype=np.float32)
    layer2_0_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer2_0_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer2_0_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer2_0_bn1_eps = 1e-5
    layer2_0_relu = None
    layer2_0_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer2_0_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer2_0_conv2_stride = 1
    layer2_0_conv2_padding = 1
    layer2_0_conv2_dilation = 1
    layer2_0_conv2_groups = 1
    layer2_0_bn2_weight = np.ones((64,), dtype=np.float32)
    layer2_0_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer2_0_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer2_0_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer2_0_bn2_eps = 1e-5
    layer2_0_downsample = None
    layer2_0_stride = 1
    layer2_1_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer2_1_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer2_1_conv1_stride = 1
    layer2_1_conv1_padding = 1
    layer2_1_conv1_dilation = 1
    layer2_1_conv1_groups = 1
    layer2_1_bn1_weight = np.ones((64,), dtype=np.float32)
    layer2_1_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer2_1_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer2_1_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer2_1_bn1_eps = 1e-5
    layer2_1_relu = None
    layer2_1_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer2_1_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer2_1_conv2_stride = 1
    layer2_1_conv2_padding = 1
    layer2_1_conv2_dilation = 1
    layer2_1_conv2_groups = 1
    layer2_1_bn2_weight = np.ones((64,), dtype=np.float32)
    layer2_1_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer2_1_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer2_1_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer2_1_bn2_eps = 1e-5
    layer2_1_downsample = None
    layer2_1_stride = 1
    layer3_0_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer3_0_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer3_0_conv1_stride = 1
    layer3_0_conv1_padding = 1
    layer3_0_conv1_dilation = 1
    layer3_0_conv1_groups = 1
    layer3_0_bn1_weight = np.ones((64,), dtype=np.float32)
    layer3_0_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer3_0_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer3_0_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer3_0_bn1_eps = 1e-5
    layer3_0_relu = None
    layer3_0_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer3_0_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer3_0_conv2_stride = 1
    layer3_0_conv2_padding = 1
    layer3_0_conv2_dilation = 1
    layer3_0_conv2_groups = 1
    layer3_0_bn2_weight = np.ones((64,), dtype=np.float32)
    layer3_0_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer3_0_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer3_0_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer3_0_bn2_eps = 1e-5
    layer3_0_downsample = None
    layer3_0_stride = 1
    layer3_1_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer3_1_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer3_1_conv1_stride = 1
    layer3_1_conv1_padding = 1
    layer3_1_conv1_dilation = 1
    layer3_1_conv1_groups = 1
    layer3_1_bn1_weight = np.ones((64,), dtype=np.float32)
    layer3_1_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer3_1_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer3_1_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer3_1_bn1_eps = 1e-5
    layer3_1_relu = None
    layer3_1_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer3_1_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer3_1_conv2_stride = 1
    layer3_1_conv2_padding = 1
    layer3_1_conv2_dilation = 1
    layer3_1_conv2_groups = 1
    layer3_1_bn2_weight = np.ones((64,), dtype=np.float32)
    layer3_1_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer3_1_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer3_1_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer3_1_bn2_eps = 1e-5
    layer3_1_downsample = None
    layer3_1_stride = 1
    layer4_0_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer4_0_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer4_0_conv1_stride = 1
    layer4_0_conv1_padding = 1
    layer4_0_conv1_dilation = 1
    layer4_0_conv1_groups = 1
    layer4_0_bn1_weight = np.ones((64,), dtype=np.float32)
    layer4_0_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer4_0_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer4_0_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer4_0_bn1_eps = 1e-5
    layer4_0_relu = None
    layer4_0_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer4_0_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer4_0_conv2_stride = 1
    layer4_0_conv2_padding = 1
    layer4_0_conv2_dilation = 1
    layer4_0_conv2_groups = 1
    layer4_0_bn2_weight = np.ones((64,), dtype=np.float32)
    layer4_0_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer4_0_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer4_0_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer4_0_bn2_eps = 1e-5
    layer4_0_downsample = None
    layer4_0_stride = 1
    layer4_1_conv1_weight = np.zeros((64, in_channels // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer4_1_conv1_bias = np.zeros((64,), dtype=np.float32)
    layer4_1_conv1_stride = 1
    layer4_1_conv1_padding = 1
    layer4_1_conv1_dilation = 1
    layer4_1_conv1_groups = 1
    layer4_1_bn1_weight = np.ones((64,), dtype=np.float32)
    layer4_1_bn1_bias = np.zeros((64,), dtype=np.float32)
    layer4_1_bn1_running_mean = np.zeros((64,), dtype=np.float32)
    layer4_1_bn1_running_var = np.ones((64,), dtype=np.float32)
    layer4_1_bn1_eps = 1e-5
    layer4_1_relu = None
    layer4_1_conv2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    layer4_1_conv2_bias = np.zeros((64,), dtype=np.float32)
    layer4_1_conv2_stride = 1
    layer4_1_conv2_padding = 1
    layer4_1_conv2_dilation = 1
    layer4_1_conv2_groups = 1
    layer4_1_bn2_weight = np.ones((64,), dtype=np.float32)
    layer4_1_bn2_bias = np.zeros((64,), dtype=np.float32)
    layer4_1_bn2_running_mean = np.zeros((64,), dtype=np.float32)
    layer4_1_bn2_running_var = np.ones((64,), dtype=np.float32)
    layer4_1_bn2_eps = 1e-5
    layer4_1_downsample = None
    layer4_1_stride = 1
    avgpool_output_size = (1, 1)
    fc_weight = np.zeros((num_classes, 512 * BasicBlock.expansion), dtype=np.float32)
    fc_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000):
    x = _conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups)
    x = _batch_norm(x, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps)
    x = np.maximum(x, 0)
    x = _maxpool2d(x, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    x = _layer1_1_forward(_layer1_0_forward(x))
    x = _layer2_1_forward(_layer2_0_forward(x))
    x = _layer3_1_forward(_layer3_0_forward(x))
    x = _layer4_1_forward(_layer4_0_forward(x))
    x = _adaptive_avg_pool2d(x, avgpool_output_size)
    x = np.reshape(x, (x.shape[0], -1))
    x = ((x) @ fc_weight.T + fc_bias)
    return x

