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

def init(num_classes=1000):
    global conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps, relu, mbconv1_0_weight, mbconv1_0_bias, mbconv1_0_stride, mbconv1_0_padding, mbconv1_0_dilation, mbconv1_0_groups, mbconv1_1_weight, mbconv1_1_bias, mbconv1_1_running_mean, mbconv1_1_running_var, mbconv1_1_eps, mbconv1_2, mbconv1_3_output_size, mbconv1_4_weight, mbconv1_4_bias, mbconv1_4_stride, mbconv1_4_padding, mbconv1_4_dilation, mbconv1_4_groups, mbconv1_5, mbconv1_6_weight, mbconv1_6_bias, mbconv1_6_stride, mbconv1_6_padding, mbconv1_6_dilation, mbconv1_6_groups, mbconv1_7, mbconv1_8_weight, mbconv1_8_bias, mbconv1_8_stride, mbconv1_8_padding, mbconv1_8_dilation, mbconv1_8_groups, mbconv1_9_weight, mbconv1_9_bias, mbconv1_9_running_mean, mbconv1_9_running_var, mbconv1_9_eps, mbconv2_0_weight, mbconv2_0_bias, mbconv2_0_stride, mbconv2_0_padding, mbconv2_0_dilation, mbconv2_0_groups, mbconv2_1_weight, mbconv2_1_bias, mbconv2_1_running_mean, mbconv2_1_running_var, mbconv2_1_eps, mbconv2_2, mbconv2_3_output_size, mbconv2_4_weight, mbconv2_4_bias, mbconv2_4_stride, mbconv2_4_padding, mbconv2_4_dilation, mbconv2_4_groups, mbconv2_5, mbconv2_6_weight, mbconv2_6_bias, mbconv2_6_stride, mbconv2_6_padding, mbconv2_6_dilation, mbconv2_6_groups, mbconv2_7, mbconv2_8_weight, mbconv2_8_bias, mbconv2_8_stride, mbconv2_8_padding, mbconv2_8_dilation, mbconv2_8_groups, mbconv2_9_weight, mbconv2_9_bias, mbconv2_9_running_mean, mbconv2_9_running_var, mbconv2_9_eps, mbconv3_0_weight, mbconv3_0_bias, mbconv3_0_stride, mbconv3_0_padding, mbconv3_0_dilation, mbconv3_0_groups, mbconv3_1_weight, mbconv3_1_bias, mbconv3_1_running_mean, mbconv3_1_running_var, mbconv3_1_eps, mbconv3_2, mbconv3_3_output_size, mbconv3_4_weight, mbconv3_4_bias, mbconv3_4_stride, mbconv3_4_padding, mbconv3_4_dilation, mbconv3_4_groups, mbconv3_5, mbconv3_6_weight, mbconv3_6_bias, mbconv3_6_stride, mbconv3_6_padding, mbconv3_6_dilation, mbconv3_6_groups, mbconv3_7, mbconv3_8_weight, mbconv3_8_bias, mbconv3_8_stride, mbconv3_8_padding, mbconv3_8_dilation, mbconv3_8_groups, mbconv3_9_weight, mbconv3_9_bias, mbconv3_9_running_mean, mbconv3_9_running_var, mbconv3_9_eps, mbconv4_0_weight, mbconv4_0_bias, mbconv4_0_stride, mbconv4_0_padding, mbconv4_0_dilation, mbconv4_0_groups, mbconv4_1_weight, mbconv4_1_bias, mbconv4_1_running_mean, mbconv4_1_running_var, mbconv4_1_eps, mbconv4_2, mbconv4_3_output_size, mbconv4_4_weight, mbconv4_4_bias, mbconv4_4_stride, mbconv4_4_padding, mbconv4_4_dilation, mbconv4_4_groups, mbconv4_5, mbconv4_6_weight, mbconv4_6_bias, mbconv4_6_stride, mbconv4_6_padding, mbconv4_6_dilation, mbconv4_6_groups, mbconv4_7, mbconv4_8_weight, mbconv4_8_bias, mbconv4_8_stride, mbconv4_8_padding, mbconv4_8_dilation, mbconv4_8_groups, mbconv4_9_weight, mbconv4_9_bias, mbconv4_9_running_mean, mbconv4_9_running_var, mbconv4_9_eps, mbconv5_0_weight, mbconv5_0_bias, mbconv5_0_stride, mbconv5_0_padding, mbconv5_0_dilation, mbconv5_0_groups, mbconv5_1_weight, mbconv5_1_bias, mbconv5_1_running_mean, mbconv5_1_running_var, mbconv5_1_eps, mbconv5_2, mbconv5_3_output_size, mbconv5_4_weight, mbconv5_4_bias, mbconv5_4_stride, mbconv5_4_padding, mbconv5_4_dilation, mbconv5_4_groups, mbconv5_5, mbconv5_6_weight, mbconv5_6_bias, mbconv5_6_stride, mbconv5_6_padding, mbconv5_6_dilation, mbconv5_6_groups, mbconv5_7, mbconv5_8_weight, mbconv5_8_bias, mbconv5_8_stride, mbconv5_8_padding, mbconv5_8_dilation, mbconv5_8_groups, mbconv5_9_weight, mbconv5_9_bias, mbconv5_9_running_mean, mbconv5_9_running_var, mbconv5_9_eps, conv_final_weight, conv_final_bias, conv_final_stride, conv_final_padding, conv_final_dilation, conv_final_groups, bn_final_weight, bn_final_bias, bn_final_running_mean, bn_final_running_var, bn_final_eps, avgpool_output_size, fc_weight, fc_bias
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
    relu = None
    mbconv1_0_weight = np.zeros((32 * 3, 32 * 3 // 32 * 3) + _as_tuple(3, 2), dtype=np.float32)
    mbconv1_0_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv1_0_stride = 1
    mbconv1_0_padding = 1
    mbconv1_0_dilation = 1
    mbconv1_0_groups = 32 * 3
    mbconv1_1_weight = np.ones((32 * 3,), dtype=np.float32)
    mbconv1_1_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv1_1_running_mean = np.zeros((32 * 3,), dtype=np.float32)
    mbconv1_1_running_var = np.ones((32 * 3,), dtype=np.float32)
    mbconv1_1_eps = 1e-5
    mbconv1_2 = None
    mbconv1_3_output_size = (1, 1)
    mbconv1_4_weight = np.zeros((32 * 3 // 4, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv1_4_bias = np.zeros((32 * 3 // 4,), dtype=np.float32)
    mbconv1_4_stride = 1
    mbconv1_4_padding = 0
    mbconv1_4_dilation = 1
    mbconv1_4_groups = 1
    mbconv1_5 = None
    mbconv1_6_weight = np.zeros((32 * 3, 32 * 3 // 4 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv1_6_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv1_6_stride = 1
    mbconv1_6_padding = 0
    mbconv1_6_dilation = 1
    mbconv1_6_groups = 1
    mbconv1_7 = None
    mbconv1_8_weight = np.zeros((96, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv1_8_bias = np.zeros((96,), dtype=np.float32)
    mbconv1_8_stride = 1
    mbconv1_8_padding = 0
    mbconv1_8_dilation = 1
    mbconv1_8_groups = 1
    mbconv1_9_weight = np.ones((96,), dtype=np.float32)
    mbconv1_9_bias = np.zeros((96,), dtype=np.float32)
    mbconv1_9_running_mean = np.zeros((96,), dtype=np.float32)
    mbconv1_9_running_var = np.ones((96,), dtype=np.float32)
    mbconv1_9_eps = 1e-5
    mbconv2_0_weight = np.zeros((32 * 3, 32 * 3 // 32 * 3) + _as_tuple(3, 2), dtype=np.float32)
    mbconv2_0_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv2_0_stride = 1
    mbconv2_0_padding = 1
    mbconv2_0_dilation = 1
    mbconv2_0_groups = 32 * 3
    mbconv2_1_weight = np.ones((32 * 3,), dtype=np.float32)
    mbconv2_1_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv2_1_running_mean = np.zeros((32 * 3,), dtype=np.float32)
    mbconv2_1_running_var = np.ones((32 * 3,), dtype=np.float32)
    mbconv2_1_eps = 1e-5
    mbconv2_2 = None
    mbconv2_3_output_size = (1, 1)
    mbconv2_4_weight = np.zeros((32 * 3 // 4, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv2_4_bias = np.zeros((32 * 3 // 4,), dtype=np.float32)
    mbconv2_4_stride = 1
    mbconv2_4_padding = 0
    mbconv2_4_dilation = 1
    mbconv2_4_groups = 1
    mbconv2_5 = None
    mbconv2_6_weight = np.zeros((32 * 3, 32 * 3 // 4 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv2_6_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv2_6_stride = 1
    mbconv2_6_padding = 0
    mbconv2_6_dilation = 1
    mbconv2_6_groups = 1
    mbconv2_7 = None
    mbconv2_8_weight = np.zeros((96, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv2_8_bias = np.zeros((96,), dtype=np.float32)
    mbconv2_8_stride = 1
    mbconv2_8_padding = 0
    mbconv2_8_dilation = 1
    mbconv2_8_groups = 1
    mbconv2_9_weight = np.ones((96,), dtype=np.float32)
    mbconv2_9_bias = np.zeros((96,), dtype=np.float32)
    mbconv2_9_running_mean = np.zeros((96,), dtype=np.float32)
    mbconv2_9_running_var = np.ones((96,), dtype=np.float32)
    mbconv2_9_eps = 1e-5
    mbconv3_0_weight = np.zeros((32 * 3, 32 * 3 // 32 * 3) + _as_tuple(3, 2), dtype=np.float32)
    mbconv3_0_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv3_0_stride = 1
    mbconv3_0_padding = 1
    mbconv3_0_dilation = 1
    mbconv3_0_groups = 32 * 3
    mbconv3_1_weight = np.ones((32 * 3,), dtype=np.float32)
    mbconv3_1_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv3_1_running_mean = np.zeros((32 * 3,), dtype=np.float32)
    mbconv3_1_running_var = np.ones((32 * 3,), dtype=np.float32)
    mbconv3_1_eps = 1e-5
    mbconv3_2 = None
    mbconv3_3_output_size = (1, 1)
    mbconv3_4_weight = np.zeros((32 * 3 // 4, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv3_4_bias = np.zeros((32 * 3 // 4,), dtype=np.float32)
    mbconv3_4_stride = 1
    mbconv3_4_padding = 0
    mbconv3_4_dilation = 1
    mbconv3_4_groups = 1
    mbconv3_5 = None
    mbconv3_6_weight = np.zeros((32 * 3, 32 * 3 // 4 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv3_6_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv3_6_stride = 1
    mbconv3_6_padding = 0
    mbconv3_6_dilation = 1
    mbconv3_6_groups = 1
    mbconv3_7 = None
    mbconv3_8_weight = np.zeros((96, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv3_8_bias = np.zeros((96,), dtype=np.float32)
    mbconv3_8_stride = 1
    mbconv3_8_padding = 0
    mbconv3_8_dilation = 1
    mbconv3_8_groups = 1
    mbconv3_9_weight = np.ones((96,), dtype=np.float32)
    mbconv3_9_bias = np.zeros((96,), dtype=np.float32)
    mbconv3_9_running_mean = np.zeros((96,), dtype=np.float32)
    mbconv3_9_running_var = np.ones((96,), dtype=np.float32)
    mbconv3_9_eps = 1e-5
    mbconv4_0_weight = np.zeros((32 * 3, 32 * 3 // 32 * 3) + _as_tuple(3, 2), dtype=np.float32)
    mbconv4_0_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv4_0_stride = 1
    mbconv4_0_padding = 1
    mbconv4_0_dilation = 1
    mbconv4_0_groups = 32 * 3
    mbconv4_1_weight = np.ones((32 * 3,), dtype=np.float32)
    mbconv4_1_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv4_1_running_mean = np.zeros((32 * 3,), dtype=np.float32)
    mbconv4_1_running_var = np.ones((32 * 3,), dtype=np.float32)
    mbconv4_1_eps = 1e-5
    mbconv4_2 = None
    mbconv4_3_output_size = (1, 1)
    mbconv4_4_weight = np.zeros((32 * 3 // 4, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv4_4_bias = np.zeros((32 * 3 // 4,), dtype=np.float32)
    mbconv4_4_stride = 1
    mbconv4_4_padding = 0
    mbconv4_4_dilation = 1
    mbconv4_4_groups = 1
    mbconv4_5 = None
    mbconv4_6_weight = np.zeros((32 * 3, 32 * 3 // 4 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv4_6_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv4_6_stride = 1
    mbconv4_6_padding = 0
    mbconv4_6_dilation = 1
    mbconv4_6_groups = 1
    mbconv4_7 = None
    mbconv4_8_weight = np.zeros((96, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv4_8_bias = np.zeros((96,), dtype=np.float32)
    mbconv4_8_stride = 1
    mbconv4_8_padding = 0
    mbconv4_8_dilation = 1
    mbconv4_8_groups = 1
    mbconv4_9_weight = np.ones((96,), dtype=np.float32)
    mbconv4_9_bias = np.zeros((96,), dtype=np.float32)
    mbconv4_9_running_mean = np.zeros((96,), dtype=np.float32)
    mbconv4_9_running_var = np.ones((96,), dtype=np.float32)
    mbconv4_9_eps = 1e-5
    mbconv5_0_weight = np.zeros((32 * 3, 32 * 3 // 32 * 3) + _as_tuple(3, 2), dtype=np.float32)
    mbconv5_0_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv5_0_stride = 1
    mbconv5_0_padding = 1
    mbconv5_0_dilation = 1
    mbconv5_0_groups = 32 * 3
    mbconv5_1_weight = np.ones((32 * 3,), dtype=np.float32)
    mbconv5_1_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv5_1_running_mean = np.zeros((32 * 3,), dtype=np.float32)
    mbconv5_1_running_var = np.ones((32 * 3,), dtype=np.float32)
    mbconv5_1_eps = 1e-5
    mbconv5_2 = None
    mbconv5_3_output_size = (1, 1)
    mbconv5_4_weight = np.zeros((32 * 3 // 4, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv5_4_bias = np.zeros((32 * 3 // 4,), dtype=np.float32)
    mbconv5_4_stride = 1
    mbconv5_4_padding = 0
    mbconv5_4_dilation = 1
    mbconv5_4_groups = 1
    mbconv5_5 = None
    mbconv5_6_weight = np.zeros((32 * 3, 32 * 3 // 4 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv5_6_bias = np.zeros((32 * 3,), dtype=np.float32)
    mbconv5_6_stride = 1
    mbconv5_6_padding = 0
    mbconv5_6_dilation = 1
    mbconv5_6_groups = 1
    mbconv5_7 = None
    mbconv5_8_weight = np.zeros((96, 32 * 3 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv5_8_bias = np.zeros((96,), dtype=np.float32)
    mbconv5_8_stride = 1
    mbconv5_8_padding = 0
    mbconv5_8_dilation = 1
    mbconv5_8_groups = 1
    mbconv5_9_weight = np.ones((96,), dtype=np.float32)
    mbconv5_9_bias = np.zeros((96,), dtype=np.float32)
    mbconv5_9_running_mean = np.zeros((96,), dtype=np.float32)
    mbconv5_9_running_var = np.ones((96,), dtype=np.float32)
    mbconv5_9_eps = 1e-5
    conv_final_weight = np.zeros((1408, 384 // 1) + _as_tuple(1, 2), dtype=np.float32)
    conv_final_bias = np.zeros((1408,), dtype=np.float32)
    conv_final_stride = 1
    conv_final_padding = 0
    conv_final_dilation = 1
    conv_final_groups = 1
    bn_final_weight = np.ones((1408,), dtype=np.float32)
    bn_final_bias = np.zeros((1408,), dtype=np.float32)
    bn_final_running_mean = np.zeros((1408,), dtype=np.float32)
    bn_final_running_var = np.ones((1408,), dtype=np.float32)
    bn_final_eps = 1e-5
    avgpool_output_size = (1, 1)
    fc_weight = np.zeros((num_classes, 1408), dtype=np.float32)
    fc_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000):
    x = np.maximum(_batch_norm(_conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups), bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps), 0)
    x = _batch_norm(_conv2d((1.0 / (1.0 + np.exp(-(_conv2d(np.maximum(_conv2d(_adaptive_avg_pool2d(np.maximum(_batch_norm(_conv2d(x, mbconv1_0_weight, mbconv1_0_bias, mbconv1_0_stride, mbconv1_0_padding, mbconv1_0_dilation, mbconv1_0_groups), mbconv1_1_weight, mbconv1_1_bias, mbconv1_1_running_mean, mbconv1_1_running_var, mbconv1_1_eps), 0), mbconv1_3_output_size), mbconv1_4_weight, mbconv1_4_bias, mbconv1_4_stride, mbconv1_4_padding, mbconv1_4_dilation, mbconv1_4_groups), 0), mbconv1_6_weight, mbconv1_6_bias, mbconv1_6_stride, mbconv1_6_padding, mbconv1_6_dilation, mbconv1_6_groups))))), mbconv1_8_weight, mbconv1_8_bias, mbconv1_8_stride, mbconv1_8_padding, mbconv1_8_dilation, mbconv1_8_groups), mbconv1_9_weight, mbconv1_9_bias, mbconv1_9_running_mean, mbconv1_9_running_var, mbconv1_9_eps)
    x = _batch_norm(_conv2d((1.0 / (1.0 + np.exp(-(_conv2d(np.maximum(_conv2d(_adaptive_avg_pool2d(np.maximum(_batch_norm(_conv2d(x, mbconv2_0_weight, mbconv2_0_bias, mbconv2_0_stride, mbconv2_0_padding, mbconv2_0_dilation, mbconv2_0_groups), mbconv2_1_weight, mbconv2_1_bias, mbconv2_1_running_mean, mbconv2_1_running_var, mbconv2_1_eps), 0), mbconv2_3_output_size), mbconv2_4_weight, mbconv2_4_bias, mbconv2_4_stride, mbconv2_4_padding, mbconv2_4_dilation, mbconv2_4_groups), 0), mbconv2_6_weight, mbconv2_6_bias, mbconv2_6_stride, mbconv2_6_padding, mbconv2_6_dilation, mbconv2_6_groups))))), mbconv2_8_weight, mbconv2_8_bias, mbconv2_8_stride, mbconv2_8_padding, mbconv2_8_dilation, mbconv2_8_groups), mbconv2_9_weight, mbconv2_9_bias, mbconv2_9_running_mean, mbconv2_9_running_var, mbconv2_9_eps)
    x = _batch_norm(_conv2d((1.0 / (1.0 + np.exp(-(_conv2d(np.maximum(_conv2d(_adaptive_avg_pool2d(np.maximum(_batch_norm(_conv2d(x, mbconv3_0_weight, mbconv3_0_bias, mbconv3_0_stride, mbconv3_0_padding, mbconv3_0_dilation, mbconv3_0_groups), mbconv3_1_weight, mbconv3_1_bias, mbconv3_1_running_mean, mbconv3_1_running_var, mbconv3_1_eps), 0), mbconv3_3_output_size), mbconv3_4_weight, mbconv3_4_bias, mbconv3_4_stride, mbconv3_4_padding, mbconv3_4_dilation, mbconv3_4_groups), 0), mbconv3_6_weight, mbconv3_6_bias, mbconv3_6_stride, mbconv3_6_padding, mbconv3_6_dilation, mbconv3_6_groups))))), mbconv3_8_weight, mbconv3_8_bias, mbconv3_8_stride, mbconv3_8_padding, mbconv3_8_dilation, mbconv3_8_groups), mbconv3_9_weight, mbconv3_9_bias, mbconv3_9_running_mean, mbconv3_9_running_var, mbconv3_9_eps)
    x = _batch_norm(_conv2d((1.0 / (1.0 + np.exp(-(_conv2d(np.maximum(_conv2d(_adaptive_avg_pool2d(np.maximum(_batch_norm(_conv2d(x, mbconv4_0_weight, mbconv4_0_bias, mbconv4_0_stride, mbconv4_0_padding, mbconv4_0_dilation, mbconv4_0_groups), mbconv4_1_weight, mbconv4_1_bias, mbconv4_1_running_mean, mbconv4_1_running_var, mbconv4_1_eps), 0), mbconv4_3_output_size), mbconv4_4_weight, mbconv4_4_bias, mbconv4_4_stride, mbconv4_4_padding, mbconv4_4_dilation, mbconv4_4_groups), 0), mbconv4_6_weight, mbconv4_6_bias, mbconv4_6_stride, mbconv4_6_padding, mbconv4_6_dilation, mbconv4_6_groups))))), mbconv4_8_weight, mbconv4_8_bias, mbconv4_8_stride, mbconv4_8_padding, mbconv4_8_dilation, mbconv4_8_groups), mbconv4_9_weight, mbconv4_9_bias, mbconv4_9_running_mean, mbconv4_9_running_var, mbconv4_9_eps)
    x = _batch_norm(_conv2d((1.0 / (1.0 + np.exp(-(_conv2d(np.maximum(_conv2d(_adaptive_avg_pool2d(np.maximum(_batch_norm(_conv2d(x, mbconv5_0_weight, mbconv5_0_bias, mbconv5_0_stride, mbconv5_0_padding, mbconv5_0_dilation, mbconv5_0_groups), mbconv5_1_weight, mbconv5_1_bias, mbconv5_1_running_mean, mbconv5_1_running_var, mbconv5_1_eps), 0), mbconv5_3_output_size), mbconv5_4_weight, mbconv5_4_bias, mbconv5_4_stride, mbconv5_4_padding, mbconv5_4_dilation, mbconv5_4_groups), 0), mbconv5_6_weight, mbconv5_6_bias, mbconv5_6_stride, mbconv5_6_padding, mbconv5_6_dilation, mbconv5_6_groups))))), mbconv5_8_weight, mbconv5_8_bias, mbconv5_8_stride, mbconv5_8_padding, mbconv5_8_dilation, mbconv5_8_groups), mbconv5_9_weight, mbconv5_9_bias, mbconv5_9_running_mean, mbconv5_9_running_var, mbconv5_9_eps)
    x = np.maximum(_batch_norm(_conv2d(x, conv_final_weight, conv_final_bias, conv_final_stride, conv_final_padding, conv_final_dilation, conv_final_groups), bn_final_weight, bn_final_bias, bn_final_running_mean, bn_final_running_var, bn_final_eps), 0)
    x = _adaptive_avg_pool2d(x, avgpool_output_size)
    x = np.reshape(x, (x.shape[0], -1))
    x = ((x) @ fc_weight.T + fc_bias)
    return x

