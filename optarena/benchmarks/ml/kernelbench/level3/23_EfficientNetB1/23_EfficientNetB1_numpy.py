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
    global conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps, mbconv1_0_weight, mbconv1_0_bias, mbconv1_0_stride, mbconv1_0_padding, mbconv1_0_dilation, mbconv1_0_groups, mbconv1_1_weight, mbconv1_1_bias, mbconv1_1_running_mean, mbconv1_1_running_var, mbconv1_1_eps, mbconv1_2, mbconv1_3_weight, mbconv1_3_bias, mbconv1_3_stride, mbconv1_3_padding, mbconv1_3_dilation, mbconv1_3_groups, mbconv1_4_weight, mbconv1_4_bias, mbconv1_4_running_mean, mbconv1_4_running_var, mbconv1_4_eps, mbconv1_5, mbconv1_6_weight, mbconv1_6_bias, mbconv1_6_stride, mbconv1_6_padding, mbconv1_6_dilation, mbconv1_6_groups, mbconv1_7_weight, mbconv1_7_bias, mbconv1_7_running_mean, mbconv1_7_running_var, mbconv1_7_eps, mbconv2_0_weight, mbconv2_0_bias, mbconv2_0_stride, mbconv2_0_padding, mbconv2_0_dilation, mbconv2_0_groups, mbconv2_1_weight, mbconv2_1_bias, mbconv2_1_running_mean, mbconv2_1_running_var, mbconv2_1_eps, mbconv2_2, mbconv2_3_weight, mbconv2_3_bias, mbconv2_3_stride, mbconv2_3_padding, mbconv2_3_dilation, mbconv2_3_groups, mbconv2_4_weight, mbconv2_4_bias, mbconv2_4_running_mean, mbconv2_4_running_var, mbconv2_4_eps, mbconv2_5, mbconv2_6_weight, mbconv2_6_bias, mbconv2_6_stride, mbconv2_6_padding, mbconv2_6_dilation, mbconv2_6_groups, mbconv2_7_weight, mbconv2_7_bias, mbconv2_7_running_mean, mbconv2_7_running_var, mbconv2_7_eps, mbconv3_0_weight, mbconv3_0_bias, mbconv3_0_stride, mbconv3_0_padding, mbconv3_0_dilation, mbconv3_0_groups, mbconv3_1_weight, mbconv3_1_bias, mbconv3_1_running_mean, mbconv3_1_running_var, mbconv3_1_eps, mbconv3_2, mbconv3_3_weight, mbconv3_3_bias, mbconv3_3_stride, mbconv3_3_padding, mbconv3_3_dilation, mbconv3_3_groups, mbconv3_4_weight, mbconv3_4_bias, mbconv3_4_running_mean, mbconv3_4_running_var, mbconv3_4_eps, mbconv3_5, mbconv3_6_weight, mbconv3_6_bias, mbconv3_6_stride, mbconv3_6_padding, mbconv3_6_dilation, mbconv3_6_groups, mbconv3_7_weight, mbconv3_7_bias, mbconv3_7_running_mean, mbconv3_7_running_var, mbconv3_7_eps, mbconv4_0_weight, mbconv4_0_bias, mbconv4_0_stride, mbconv4_0_padding, mbconv4_0_dilation, mbconv4_0_groups, mbconv4_1_weight, mbconv4_1_bias, mbconv4_1_running_mean, mbconv4_1_running_var, mbconv4_1_eps, mbconv4_2, mbconv4_3_weight, mbconv4_3_bias, mbconv4_3_stride, mbconv4_3_padding, mbconv4_3_dilation, mbconv4_3_groups, mbconv4_4_weight, mbconv4_4_bias, mbconv4_4_running_mean, mbconv4_4_running_var, mbconv4_4_eps, mbconv4_5, mbconv4_6_weight, mbconv4_6_bias, mbconv4_6_stride, mbconv4_6_padding, mbconv4_6_dilation, mbconv4_6_groups, mbconv4_7_weight, mbconv4_7_bias, mbconv4_7_running_mean, mbconv4_7_running_var, mbconv4_7_eps, mbconv5_0_weight, mbconv5_0_bias, mbconv5_0_stride, mbconv5_0_padding, mbconv5_0_dilation, mbconv5_0_groups, mbconv5_1_weight, mbconv5_1_bias, mbconv5_1_running_mean, mbconv5_1_running_var, mbconv5_1_eps, mbconv5_2, mbconv5_3_weight, mbconv5_3_bias, mbconv5_3_stride, mbconv5_3_padding, mbconv5_3_dilation, mbconv5_3_groups, mbconv5_4_weight, mbconv5_4_bias, mbconv5_4_running_mean, mbconv5_4_running_var, mbconv5_4_eps, mbconv5_5, mbconv5_6_weight, mbconv5_6_bias, mbconv5_6_stride, mbconv5_6_padding, mbconv5_6_dilation, mbconv5_6_groups, mbconv5_7_weight, mbconv5_7_bias, mbconv5_7_running_mean, mbconv5_7_running_var, mbconv5_7_eps, mbconv6_0_weight, mbconv6_0_bias, mbconv6_0_stride, mbconv6_0_padding, mbconv6_0_dilation, mbconv6_0_groups, mbconv6_1_weight, mbconv6_1_bias, mbconv6_1_running_mean, mbconv6_1_running_var, mbconv6_1_eps, mbconv6_2, mbconv6_3_weight, mbconv6_3_bias, mbconv6_3_stride, mbconv6_3_padding, mbconv6_3_dilation, mbconv6_3_groups, mbconv6_4_weight, mbconv6_4_bias, mbconv6_4_running_mean, mbconv6_4_running_var, mbconv6_4_eps, mbconv6_5, mbconv6_6_weight, mbconv6_6_bias, mbconv6_6_stride, mbconv6_6_padding, mbconv6_6_dilation, mbconv6_6_groups, mbconv6_7_weight, mbconv6_7_bias, mbconv6_7_running_mean, mbconv6_7_running_var, mbconv6_7_eps, mbconv7_0_weight, mbconv7_0_bias, mbconv7_0_stride, mbconv7_0_padding, mbconv7_0_dilation, mbconv7_0_groups, mbconv7_1_weight, mbconv7_1_bias, mbconv7_1_running_mean, mbconv7_1_running_var, mbconv7_1_eps, mbconv7_2, mbconv7_3_weight, mbconv7_3_bias, mbconv7_3_stride, mbconv7_3_padding, mbconv7_3_dilation, mbconv7_3_groups, mbconv7_4_weight, mbconv7_4_bias, mbconv7_4_running_mean, mbconv7_4_running_var, mbconv7_4_eps, mbconv7_5, mbconv7_6_weight, mbconv7_6_bias, mbconv7_6_stride, mbconv7_6_padding, mbconv7_6_dilation, mbconv7_6_groups, mbconv7_7_weight, mbconv7_7_bias, mbconv7_7_running_mean, mbconv7_7_running_var, mbconv7_7_eps, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, bn2_eps, fc_weight, fc_bias
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
    mbconv1_0_weight = np.zeros((round(32 * 1), 32 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv1_0_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv1_0_stride = 1
    mbconv1_0_padding = 0
    mbconv1_0_dilation = 1
    mbconv1_0_groups = 1
    mbconv1_1_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv1_1_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv1_1_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv1_1_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv1_1_eps = 1e-5
    mbconv1_2 = None
    mbconv1_3_weight = np.zeros((round(32 * 1), round(32 * 1) // round(32 * 1)) + _as_tuple(3, 2), dtype=np.float32)
    mbconv1_3_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv1_3_stride = 1
    mbconv1_3_padding = 1
    mbconv1_3_dilation = 1
    mbconv1_3_groups = round(32 * 1)
    mbconv1_4_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv1_4_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv1_4_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv1_4_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv1_4_eps = 1e-5
    mbconv1_5 = None
    mbconv1_6_weight = np.zeros((16, round(32 * 1) // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv1_6_bias = np.zeros((16,), dtype=np.float32)
    mbconv1_6_stride = 1
    mbconv1_6_padding = 0
    mbconv1_6_dilation = 1
    mbconv1_6_groups = 1
    mbconv1_7_weight = np.ones((16,), dtype=np.float32)
    mbconv1_7_bias = np.zeros((16,), dtype=np.float32)
    mbconv1_7_running_mean = np.zeros((16,), dtype=np.float32)
    mbconv1_7_running_var = np.ones((16,), dtype=np.float32)
    mbconv1_7_eps = 1e-5
    mbconv2_0_weight = np.zeros((round(32 * 1), 32 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv2_0_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv2_0_stride = 1
    mbconv2_0_padding = 0
    mbconv2_0_dilation = 1
    mbconv2_0_groups = 1
    mbconv2_1_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv2_1_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv2_1_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv2_1_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv2_1_eps = 1e-5
    mbconv2_2 = None
    mbconv2_3_weight = np.zeros((round(32 * 1), round(32 * 1) // round(32 * 1)) + _as_tuple(3, 2), dtype=np.float32)
    mbconv2_3_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv2_3_stride = 1
    mbconv2_3_padding = 1
    mbconv2_3_dilation = 1
    mbconv2_3_groups = round(32 * 1)
    mbconv2_4_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv2_4_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv2_4_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv2_4_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv2_4_eps = 1e-5
    mbconv2_5 = None
    mbconv2_6_weight = np.zeros((16, round(32 * 1) // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv2_6_bias = np.zeros((16,), dtype=np.float32)
    mbconv2_6_stride = 1
    mbconv2_6_padding = 0
    mbconv2_6_dilation = 1
    mbconv2_6_groups = 1
    mbconv2_7_weight = np.ones((16,), dtype=np.float32)
    mbconv2_7_bias = np.zeros((16,), dtype=np.float32)
    mbconv2_7_running_mean = np.zeros((16,), dtype=np.float32)
    mbconv2_7_running_var = np.ones((16,), dtype=np.float32)
    mbconv2_7_eps = 1e-5
    mbconv3_0_weight = np.zeros((round(32 * 1), 32 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv3_0_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv3_0_stride = 1
    mbconv3_0_padding = 0
    mbconv3_0_dilation = 1
    mbconv3_0_groups = 1
    mbconv3_1_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv3_1_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv3_1_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv3_1_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv3_1_eps = 1e-5
    mbconv3_2 = None
    mbconv3_3_weight = np.zeros((round(32 * 1), round(32 * 1) // round(32 * 1)) + _as_tuple(3, 2), dtype=np.float32)
    mbconv3_3_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv3_3_stride = 1
    mbconv3_3_padding = 1
    mbconv3_3_dilation = 1
    mbconv3_3_groups = round(32 * 1)
    mbconv3_4_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv3_4_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv3_4_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv3_4_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv3_4_eps = 1e-5
    mbconv3_5 = None
    mbconv3_6_weight = np.zeros((16, round(32 * 1) // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv3_6_bias = np.zeros((16,), dtype=np.float32)
    mbconv3_6_stride = 1
    mbconv3_6_padding = 0
    mbconv3_6_dilation = 1
    mbconv3_6_groups = 1
    mbconv3_7_weight = np.ones((16,), dtype=np.float32)
    mbconv3_7_bias = np.zeros((16,), dtype=np.float32)
    mbconv3_7_running_mean = np.zeros((16,), dtype=np.float32)
    mbconv3_7_running_var = np.ones((16,), dtype=np.float32)
    mbconv3_7_eps = 1e-5
    mbconv4_0_weight = np.zeros((round(32 * 1), 32 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv4_0_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv4_0_stride = 1
    mbconv4_0_padding = 0
    mbconv4_0_dilation = 1
    mbconv4_0_groups = 1
    mbconv4_1_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv4_1_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv4_1_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv4_1_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv4_1_eps = 1e-5
    mbconv4_2 = None
    mbconv4_3_weight = np.zeros((round(32 * 1), round(32 * 1) // round(32 * 1)) + _as_tuple(3, 2), dtype=np.float32)
    mbconv4_3_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv4_3_stride = 1
    mbconv4_3_padding = 1
    mbconv4_3_dilation = 1
    mbconv4_3_groups = round(32 * 1)
    mbconv4_4_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv4_4_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv4_4_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv4_4_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv4_4_eps = 1e-5
    mbconv4_5 = None
    mbconv4_6_weight = np.zeros((16, round(32 * 1) // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv4_6_bias = np.zeros((16,), dtype=np.float32)
    mbconv4_6_stride = 1
    mbconv4_6_padding = 0
    mbconv4_6_dilation = 1
    mbconv4_6_groups = 1
    mbconv4_7_weight = np.ones((16,), dtype=np.float32)
    mbconv4_7_bias = np.zeros((16,), dtype=np.float32)
    mbconv4_7_running_mean = np.zeros((16,), dtype=np.float32)
    mbconv4_7_running_var = np.ones((16,), dtype=np.float32)
    mbconv4_7_eps = 1e-5
    mbconv5_0_weight = np.zeros((round(32 * 1), 32 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv5_0_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv5_0_stride = 1
    mbconv5_0_padding = 0
    mbconv5_0_dilation = 1
    mbconv5_0_groups = 1
    mbconv5_1_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv5_1_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv5_1_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv5_1_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv5_1_eps = 1e-5
    mbconv5_2 = None
    mbconv5_3_weight = np.zeros((round(32 * 1), round(32 * 1) // round(32 * 1)) + _as_tuple(3, 2), dtype=np.float32)
    mbconv5_3_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv5_3_stride = 1
    mbconv5_3_padding = 1
    mbconv5_3_dilation = 1
    mbconv5_3_groups = round(32 * 1)
    mbconv5_4_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv5_4_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv5_4_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv5_4_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv5_4_eps = 1e-5
    mbconv5_5 = None
    mbconv5_6_weight = np.zeros((16, round(32 * 1) // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv5_6_bias = np.zeros((16,), dtype=np.float32)
    mbconv5_6_stride = 1
    mbconv5_6_padding = 0
    mbconv5_6_dilation = 1
    mbconv5_6_groups = 1
    mbconv5_7_weight = np.ones((16,), dtype=np.float32)
    mbconv5_7_bias = np.zeros((16,), dtype=np.float32)
    mbconv5_7_running_mean = np.zeros((16,), dtype=np.float32)
    mbconv5_7_running_var = np.ones((16,), dtype=np.float32)
    mbconv5_7_eps = 1e-5
    mbconv6_0_weight = np.zeros((round(32 * 1), 32 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv6_0_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv6_0_stride = 1
    mbconv6_0_padding = 0
    mbconv6_0_dilation = 1
    mbconv6_0_groups = 1
    mbconv6_1_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv6_1_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv6_1_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv6_1_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv6_1_eps = 1e-5
    mbconv6_2 = None
    mbconv6_3_weight = np.zeros((round(32 * 1), round(32 * 1) // round(32 * 1)) + _as_tuple(3, 2), dtype=np.float32)
    mbconv6_3_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv6_3_stride = 1
    mbconv6_3_padding = 1
    mbconv6_3_dilation = 1
    mbconv6_3_groups = round(32 * 1)
    mbconv6_4_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv6_4_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv6_4_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv6_4_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv6_4_eps = 1e-5
    mbconv6_5 = None
    mbconv6_6_weight = np.zeros((16, round(32 * 1) // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv6_6_bias = np.zeros((16,), dtype=np.float32)
    mbconv6_6_stride = 1
    mbconv6_6_padding = 0
    mbconv6_6_dilation = 1
    mbconv6_6_groups = 1
    mbconv6_7_weight = np.ones((16,), dtype=np.float32)
    mbconv6_7_bias = np.zeros((16,), dtype=np.float32)
    mbconv6_7_running_mean = np.zeros((16,), dtype=np.float32)
    mbconv6_7_running_var = np.ones((16,), dtype=np.float32)
    mbconv6_7_eps = 1e-5
    mbconv7_0_weight = np.zeros((round(32 * 1), 32 // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv7_0_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv7_0_stride = 1
    mbconv7_0_padding = 0
    mbconv7_0_dilation = 1
    mbconv7_0_groups = 1
    mbconv7_1_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv7_1_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv7_1_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv7_1_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv7_1_eps = 1e-5
    mbconv7_2 = None
    mbconv7_3_weight = np.zeros((round(32 * 1), round(32 * 1) // round(32 * 1)) + _as_tuple(3, 2), dtype=np.float32)
    mbconv7_3_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv7_3_stride = 1
    mbconv7_3_padding = 1
    mbconv7_3_dilation = 1
    mbconv7_3_groups = round(32 * 1)
    mbconv7_4_weight = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv7_4_bias = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv7_4_running_mean = np.zeros((round(32 * 1),), dtype=np.float32)
    mbconv7_4_running_var = np.ones((round(32 * 1),), dtype=np.float32)
    mbconv7_4_eps = 1e-5
    mbconv7_5 = None
    mbconv7_6_weight = np.zeros((16, round(32 * 1) // 1) + _as_tuple(1, 2), dtype=np.float32)
    mbconv7_6_bias = np.zeros((16,), dtype=np.float32)
    mbconv7_6_stride = 1
    mbconv7_6_padding = 0
    mbconv7_6_dilation = 1
    mbconv7_6_groups = 1
    mbconv7_7_weight = np.ones((16,), dtype=np.float32)
    mbconv7_7_bias = np.zeros((16,), dtype=np.float32)
    mbconv7_7_running_mean = np.zeros((16,), dtype=np.float32)
    mbconv7_7_running_var = np.ones((16,), dtype=np.float32)
    mbconv7_7_eps = 1e-5
    conv2_weight = np.zeros((1280, 320 // 1) + _as_tuple(1, 2), dtype=np.float32)
    conv2_bias = np.zeros((1280,), dtype=np.float32)
    conv2_stride = 1
    conv2_padding = 0
    conv2_dilation = 1
    conv2_groups = 1
    bn2_weight = np.ones((1280,), dtype=np.float32)
    bn2_bias = np.zeros((1280,), dtype=np.float32)
    bn2_running_mean = np.zeros((1280,), dtype=np.float32)
    bn2_running_var = np.ones((1280,), dtype=np.float32)
    bn2_eps = 1e-5
    fc_weight = np.zeros((num_classes, 1280), dtype=np.float32)
    fc_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000):
    x = np.maximum(_batch_norm(_conv2d(x, conv1_weight, conv1_bias, conv1_stride, conv1_padding, conv1_dilation, conv1_groups), bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, bn1_eps), 0)
    x = _batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(x, mbconv1_0_weight, mbconv1_0_bias, mbconv1_0_stride, mbconv1_0_padding, mbconv1_0_dilation, mbconv1_0_groups), mbconv1_1_weight, mbconv1_1_bias, mbconv1_1_running_mean, mbconv1_1_running_var, mbconv1_1_eps), 0.0, 6.0), mbconv1_3_weight, mbconv1_3_bias, mbconv1_3_stride, mbconv1_3_padding, mbconv1_3_dilation, mbconv1_3_groups), mbconv1_4_weight, mbconv1_4_bias, mbconv1_4_running_mean, mbconv1_4_running_var, mbconv1_4_eps), 0.0, 6.0), mbconv1_6_weight, mbconv1_6_bias, mbconv1_6_stride, mbconv1_6_padding, mbconv1_6_dilation, mbconv1_6_groups), mbconv1_7_weight, mbconv1_7_bias, mbconv1_7_running_mean, mbconv1_7_running_var, mbconv1_7_eps)
    x = _batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(x, mbconv2_0_weight, mbconv2_0_bias, mbconv2_0_stride, mbconv2_0_padding, mbconv2_0_dilation, mbconv2_0_groups), mbconv2_1_weight, mbconv2_1_bias, mbconv2_1_running_mean, mbconv2_1_running_var, mbconv2_1_eps), 0.0, 6.0), mbconv2_3_weight, mbconv2_3_bias, mbconv2_3_stride, mbconv2_3_padding, mbconv2_3_dilation, mbconv2_3_groups), mbconv2_4_weight, mbconv2_4_bias, mbconv2_4_running_mean, mbconv2_4_running_var, mbconv2_4_eps), 0.0, 6.0), mbconv2_6_weight, mbconv2_6_bias, mbconv2_6_stride, mbconv2_6_padding, mbconv2_6_dilation, mbconv2_6_groups), mbconv2_7_weight, mbconv2_7_bias, mbconv2_7_running_mean, mbconv2_7_running_var, mbconv2_7_eps)
    x = _batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(x, mbconv3_0_weight, mbconv3_0_bias, mbconv3_0_stride, mbconv3_0_padding, mbconv3_0_dilation, mbconv3_0_groups), mbconv3_1_weight, mbconv3_1_bias, mbconv3_1_running_mean, mbconv3_1_running_var, mbconv3_1_eps), 0.0, 6.0), mbconv3_3_weight, mbconv3_3_bias, mbconv3_3_stride, mbconv3_3_padding, mbconv3_3_dilation, mbconv3_3_groups), mbconv3_4_weight, mbconv3_4_bias, mbconv3_4_running_mean, mbconv3_4_running_var, mbconv3_4_eps), 0.0, 6.0), mbconv3_6_weight, mbconv3_6_bias, mbconv3_6_stride, mbconv3_6_padding, mbconv3_6_dilation, mbconv3_6_groups), mbconv3_7_weight, mbconv3_7_bias, mbconv3_7_running_mean, mbconv3_7_running_var, mbconv3_7_eps)
    x = _batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(x, mbconv4_0_weight, mbconv4_0_bias, mbconv4_0_stride, mbconv4_0_padding, mbconv4_0_dilation, mbconv4_0_groups), mbconv4_1_weight, mbconv4_1_bias, mbconv4_1_running_mean, mbconv4_1_running_var, mbconv4_1_eps), 0.0, 6.0), mbconv4_3_weight, mbconv4_3_bias, mbconv4_3_stride, mbconv4_3_padding, mbconv4_3_dilation, mbconv4_3_groups), mbconv4_4_weight, mbconv4_4_bias, mbconv4_4_running_mean, mbconv4_4_running_var, mbconv4_4_eps), 0.0, 6.0), mbconv4_6_weight, mbconv4_6_bias, mbconv4_6_stride, mbconv4_6_padding, mbconv4_6_dilation, mbconv4_6_groups), mbconv4_7_weight, mbconv4_7_bias, mbconv4_7_running_mean, mbconv4_7_running_var, mbconv4_7_eps)
    x = _batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(x, mbconv5_0_weight, mbconv5_0_bias, mbconv5_0_stride, mbconv5_0_padding, mbconv5_0_dilation, mbconv5_0_groups), mbconv5_1_weight, mbconv5_1_bias, mbconv5_1_running_mean, mbconv5_1_running_var, mbconv5_1_eps), 0.0, 6.0), mbconv5_3_weight, mbconv5_3_bias, mbconv5_3_stride, mbconv5_3_padding, mbconv5_3_dilation, mbconv5_3_groups), mbconv5_4_weight, mbconv5_4_bias, mbconv5_4_running_mean, mbconv5_4_running_var, mbconv5_4_eps), 0.0, 6.0), mbconv5_6_weight, mbconv5_6_bias, mbconv5_6_stride, mbconv5_6_padding, mbconv5_6_dilation, mbconv5_6_groups), mbconv5_7_weight, mbconv5_7_bias, mbconv5_7_running_mean, mbconv5_7_running_var, mbconv5_7_eps)
    x = _batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(x, mbconv6_0_weight, mbconv6_0_bias, mbconv6_0_stride, mbconv6_0_padding, mbconv6_0_dilation, mbconv6_0_groups), mbconv6_1_weight, mbconv6_1_bias, mbconv6_1_running_mean, mbconv6_1_running_var, mbconv6_1_eps), 0.0, 6.0), mbconv6_3_weight, mbconv6_3_bias, mbconv6_3_stride, mbconv6_3_padding, mbconv6_3_dilation, mbconv6_3_groups), mbconv6_4_weight, mbconv6_4_bias, mbconv6_4_running_mean, mbconv6_4_running_var, mbconv6_4_eps), 0.0, 6.0), mbconv6_6_weight, mbconv6_6_bias, mbconv6_6_stride, mbconv6_6_padding, mbconv6_6_dilation, mbconv6_6_groups), mbconv6_7_weight, mbconv6_7_bias, mbconv6_7_running_mean, mbconv6_7_running_var, mbconv6_7_eps)
    x = _batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(np.clip(_batch_norm(_conv2d(x, mbconv7_0_weight, mbconv7_0_bias, mbconv7_0_stride, mbconv7_0_padding, mbconv7_0_dilation, mbconv7_0_groups), mbconv7_1_weight, mbconv7_1_bias, mbconv7_1_running_mean, mbconv7_1_running_var, mbconv7_1_eps), 0.0, 6.0), mbconv7_3_weight, mbconv7_3_bias, mbconv7_3_stride, mbconv7_3_padding, mbconv7_3_dilation, mbconv7_3_groups), mbconv7_4_weight, mbconv7_4_bias, mbconv7_4_running_mean, mbconv7_4_running_var, mbconv7_4_eps), 0.0, 6.0), mbconv7_6_weight, mbconv7_6_bias, mbconv7_6_stride, mbconv7_6_padding, mbconv7_6_dilation, mbconv7_6_groups), mbconv7_7_weight, mbconv7_7_bias, mbconv7_7_running_mean, mbconv7_7_running_var, mbconv7_7_eps)
    x = np.maximum(_batch_norm(_conv2d(x, conv2_weight, conv2_bias, conv2_stride, conv2_padding, conv2_dilation, conv2_groups), bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, bn2_eps), 0)
    x = _adaptive_avg_pool2d(x, (1, 1))
    x = np.reshape(x, (x.shape[0], -1))
    x = ((x) @ fc_weight.T + fc_bias)
    return x

