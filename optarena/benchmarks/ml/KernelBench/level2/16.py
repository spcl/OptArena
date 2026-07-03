import numpy as np

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def _as_tuple(value, dims):
    if isinstance(value, tuple):
        return value
    return tuple(value for _ in range(dims))


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

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        self.conv_transpose_weight = np.zeros((in_channels, out_channels // 1) + _as_tuple(kernel_size, 2), dtype=np.float32)
        self.conv_transpose_bias = np.zeros((out_channels,), dtype=np.float32)
        self.conv_transpose_stride = stride
        self.conv_transpose_padding = padding
        self.conv_transpose_dilation = 1
        self.conv_transpose_groups = 1
        self.conv_transpose_output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = _conv_transpose2d(x, self.conv_transpose_weight, self.conv_transpose_bias, self.conv_transpose_stride, self.conv_transpose_padding, self.conv_transpose_output_padding, self.conv_transpose_dilation, self.conv_transpose_groups)
        x = ((x) * np.tanh((np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))))
        x = (x + self.add_value)
        x = np.clip(x, (-1), 1)
        x = (x * self.scale)
        return x

