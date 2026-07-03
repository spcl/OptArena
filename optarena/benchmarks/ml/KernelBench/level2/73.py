import numpy as np

batch_size = 128
in_channels = 8
out_channels = 64
height, width = (128, 128)
kernel_size = 3
scaling_factor = 2.0

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

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        self.conv_weight = np.zeros((out_channels, in_channels // 1) + _as_tuple(kernel_size, 2), dtype=np.float32)
        self.conv_bias = np.zeros((out_channels,), dtype=np.float32)
        self.conv_stride = 1
        self.conv_padding = 0
        self.conv_dilation = 1
        self.conv_groups = 1
        self.bn_weight = np.ones((out_channels,), dtype=np.float32)
        self.bn_bias = np.zeros((out_channels,), dtype=np.float32)
        self.bn_running_mean = np.zeros((out_channels,), dtype=np.float32)
        self.bn_running_var = np.ones((out_channels,), dtype=np.float32)
        self.bn_eps = 1e-5
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = _conv2d(x, self.conv_weight, self.conv_bias, self.conv_stride, self.conv_padding, self.conv_dilation, self.conv_groups)
        x = _batch_norm(x, self.bn_weight, self.bn_bias, self.bn_running_mean, self.bn_running_var, self.bn_eps)
        x = (x * self.scaling_factor)
        return x

