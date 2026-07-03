import numpy as np

batch_size = 16
in_channels = 16
out_channels = 64
depth = 32
height = width = 128
kernel_size = 3
stride = 1
padding = 1
scaling_factor = 2.0

def _as_tuple(value, dims):
    if isinstance(value, tuple):
        return value
    return tuple(value for _ in range(dims))


def _conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    if isinstance(stride, int): stride = (stride, stride, stride)
    if isinstance(padding, int): padding = (padding, padding, padding)
    if isinstance(output_padding, int): output_padding = (output_padding, output_padding, output_padding)
    if isinstance(dilation, int): dilation = (dilation, dilation, dilation)
    n, c_in, d, h, w = x.shape
    _, c_out_per_group, kd, kh, kw = weight.shape
    c_out = c_out_per_group * groups
    od = (d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kd - 1) + output_padding[0] + 1
    oh = (h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kh - 1) + output_padding[1] + 1
    ow = (w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kw - 1) + output_padding[2] + 1
    out = np.zeros((n, c_out, od, oh, ow), dtype=x.dtype)
    in_per_group = c_in // groups
    for b in range(n):
        for ic in range(c_in):
            g = ic // in_per_group
            for iz in range(d):
                for iy in range(h):
                    for ix in range(w):
                        for kz in range(kd):
                            oz = iz * stride[0] - padding[0] + kz * dilation[0]
                            if 0 <= oz < od:
                                for ky in range(kh):
                                    oy = iy * stride[1] - padding[1] + ky * dilation[1]
                                    if 0 <= oy < oh:
                                        for kx in range(kw):
                                            ox = ix * stride[2] - padding[2] + kx * dilation[2]
                                            if 0 <= ox < ow:
                                                for ocg in range(c_out_per_group):
                                                    out[b, g * c_out_per_group + ocg, oz, oy, ox] += x[b, ic, iz, iy, ix] * weight[ic, ocg, kz, ky, kx]
    out += bias.reshape(1, -1, 1, 1, 1)
    return out


def _softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        self.conv_transpose_weight = np.zeros((in_channels, out_channels // 1) + _as_tuple(kernel_size, 3), dtype=np.float32)
        self.conv_transpose_bias = np.zeros((out_channels,), dtype=np.float32)
        self.conv_transpose_stride = stride
        self.conv_transpose_padding = padding
        self.conv_transpose_dilation = 1
        self.conv_transpose_groups = 1
        self.conv_transpose_output_padding = 0
        self.bias = np.zeros(1, dtype=np.float32)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = _conv_transpose3d(x, self.conv_transpose_weight, self.conv_transpose_bias, self.conv_transpose_stride, self.conv_transpose_padding, self.conv_transpose_output_padding, self.conv_transpose_dilation, self.conv_transpose_groups)
        x = np.mean(x, axis=2, keepdims=True)
        x = (x + self.bias)
        x = _softmax(x, axis=1)
        x = np.tanh(x)
        x = (x * self.scaling_factor)
        return x

