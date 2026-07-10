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
    global features_0_weight, features_0_bias, features_0_stride, features_0_padding, features_0_dilation, features_0_groups, features_1, features_2_weight, features_2_bias, features_2_stride, features_2_padding, features_2_dilation, features_2_groups, features_3, features_4_kernel_size, features_4_stride, features_4_padding, features_5_weight, features_5_bias, features_5_stride, features_5_padding, features_5_dilation, features_5_groups, features_6, features_7_weight, features_7_bias, features_7_stride, features_7_padding, features_7_dilation, features_7_groups, features_8, features_9_kernel_size, features_9_stride, features_9_padding, features_10_weight, features_10_bias, features_10_stride, features_10_padding, features_10_dilation, features_10_groups, features_11, features_12_weight, features_12_bias, features_12_stride, features_12_padding, features_12_dilation, features_12_groups, features_13, features_14_weight, features_14_bias, features_14_stride, features_14_padding, features_14_dilation, features_14_groups, features_15, features_16_weight, features_16_bias, features_16_stride, features_16_padding, features_16_dilation, features_16_groups, features_17, features_18_kernel_size, features_18_stride, features_18_padding, features_19_weight, features_19_bias, features_19_stride, features_19_padding, features_19_dilation, features_19_groups, features_20, features_21_weight, features_21_bias, features_21_stride, features_21_padding, features_21_dilation, features_21_groups, features_22, features_23_weight, features_23_bias, features_23_stride, features_23_padding, features_23_dilation, features_23_groups, features_24, features_25_weight, features_25_bias, features_25_stride, features_25_padding, features_25_dilation, features_25_groups, features_26, features_27_kernel_size, features_27_stride, features_27_padding, features_28_weight, features_28_bias, features_28_stride, features_28_padding, features_28_dilation, features_28_groups, features_29, features_30_weight, features_30_bias, features_30_stride, features_30_padding, features_30_dilation, features_30_groups, features_31, features_32_weight, features_32_bias, features_32_stride, features_32_padding, features_32_dilation, features_32_groups, features_33, features_34_weight, features_34_bias, features_34_stride, features_34_padding, features_34_dilation, features_34_groups, features_35, features_36_kernel_size, features_36_stride, features_36_padding, classifier_0_weight, classifier_0_bias, classifier_1, classifier_2, classifier_3_weight, classifier_3_bias, classifier_4, classifier_5, classifier_6_weight, classifier_6_bias
    features_0_weight = np.zeros((64, 3 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_0_bias = np.zeros((64,), dtype=np.float32)
    features_0_stride = 1
    features_0_padding = 1
    features_0_dilation = 1
    features_0_groups = 1
    features_1 = None
    features_2_weight = np.zeros((64, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_2_bias = np.zeros((64,), dtype=np.float32)
    features_2_stride = 1
    features_2_padding = 1
    features_2_dilation = 1
    features_2_groups = 1
    features_3 = None
    features_4_kernel_size = 2
    features_4_stride = 2
    features_4_padding = 0
    features_5_weight = np.zeros((128, 64 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_5_bias = np.zeros((128,), dtype=np.float32)
    features_5_stride = 1
    features_5_padding = 1
    features_5_dilation = 1
    features_5_groups = 1
    features_6 = None
    features_7_weight = np.zeros((128, 128 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_7_bias = np.zeros((128,), dtype=np.float32)
    features_7_stride = 1
    features_7_padding = 1
    features_7_dilation = 1
    features_7_groups = 1
    features_8 = None
    features_9_kernel_size = 2
    features_9_stride = 2
    features_9_padding = 0
    features_10_weight = np.zeros((256, 128 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_10_bias = np.zeros((256,), dtype=np.float32)
    features_10_stride = 1
    features_10_padding = 1
    features_10_dilation = 1
    features_10_groups = 1
    features_11 = None
    features_12_weight = np.zeros((256, 256 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_12_bias = np.zeros((256,), dtype=np.float32)
    features_12_stride = 1
    features_12_padding = 1
    features_12_dilation = 1
    features_12_groups = 1
    features_13 = None
    features_14_weight = np.zeros((256, 256 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_14_bias = np.zeros((256,), dtype=np.float32)
    features_14_stride = 1
    features_14_padding = 1
    features_14_dilation = 1
    features_14_groups = 1
    features_15 = None
    features_16_weight = np.zeros((256, 256 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_16_bias = np.zeros((256,), dtype=np.float32)
    features_16_stride = 1
    features_16_padding = 1
    features_16_dilation = 1
    features_16_groups = 1
    features_17 = None
    features_18_kernel_size = 2
    features_18_stride = 2
    features_18_padding = 0
    features_19_weight = np.zeros((512, 256 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_19_bias = np.zeros((512,), dtype=np.float32)
    features_19_stride = 1
    features_19_padding = 1
    features_19_dilation = 1
    features_19_groups = 1
    features_20 = None
    features_21_weight = np.zeros((512, 512 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_21_bias = np.zeros((512,), dtype=np.float32)
    features_21_stride = 1
    features_21_padding = 1
    features_21_dilation = 1
    features_21_groups = 1
    features_22 = None
    features_23_weight = np.zeros((512, 512 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_23_bias = np.zeros((512,), dtype=np.float32)
    features_23_stride = 1
    features_23_padding = 1
    features_23_dilation = 1
    features_23_groups = 1
    features_24 = None
    features_25_weight = np.zeros((512, 512 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_25_bias = np.zeros((512,), dtype=np.float32)
    features_25_stride = 1
    features_25_padding = 1
    features_25_dilation = 1
    features_25_groups = 1
    features_26 = None
    features_27_kernel_size = 2
    features_27_stride = 2
    features_27_padding = 0
    features_28_weight = np.zeros((512, 512 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_28_bias = np.zeros((512,), dtype=np.float32)
    features_28_stride = 1
    features_28_padding = 1
    features_28_dilation = 1
    features_28_groups = 1
    features_29 = None
    features_30_weight = np.zeros((512, 512 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_30_bias = np.zeros((512,), dtype=np.float32)
    features_30_stride = 1
    features_30_padding = 1
    features_30_dilation = 1
    features_30_groups = 1
    features_31 = None
    features_32_weight = np.zeros((512, 512 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_32_bias = np.zeros((512,), dtype=np.float32)
    features_32_stride = 1
    features_32_padding = 1
    features_32_dilation = 1
    features_32_groups = 1
    features_33 = None
    features_34_weight = np.zeros((512, 512 // 1) + _as_tuple(3, 2), dtype=np.float32)
    features_34_bias = np.zeros((512,), dtype=np.float32)
    features_34_stride = 1
    features_34_padding = 1
    features_34_dilation = 1
    features_34_groups = 1
    features_35 = None
    features_36_kernel_size = 2
    features_36_stride = 2
    features_36_padding = 0
    classifier_0_weight = np.zeros((4096, 512 * 7 * 7), dtype=np.float32)
    classifier_0_bias = np.zeros((4096,), dtype=np.float32) if True else np.zeros((4096,), dtype=np.float32)
    classifier_1 = None
    classifier_2 = None
    classifier_3_weight = np.zeros((4096, 4096), dtype=np.float32)
    classifier_3_bias = np.zeros((4096,), dtype=np.float32) if True else np.zeros((4096,), dtype=np.float32)
    classifier_4 = None
    classifier_5 = None
    classifier_6_weight = np.zeros((num_classes, 4096), dtype=np.float32)
    classifier_6_bias = np.zeros((num_classes,), dtype=np.float32) if True else np.zeros((num_classes,), dtype=np.float32)

def forward(x, num_classes=1000):
    x = _maxpool2d(np.maximum(_conv2d(np.maximum(_conv2d(np.maximum(_conv2d(np.maximum(_conv2d(_maxpool2d(np.maximum(_conv2d(np.maximum(_conv2d(np.maximum(_conv2d(np.maximum(_conv2d(_maxpool2d(np.maximum(_conv2d(np.maximum(_conv2d(np.maximum(_conv2d(np.maximum(_conv2d(_maxpool2d(np.maximum(_conv2d(np.maximum(_conv2d(_maxpool2d(np.maximum(_conv2d(np.maximum(_conv2d(x, features_0_weight, features_0_bias, features_0_stride, features_0_padding, features_0_dilation, features_0_groups), 0), features_2_weight, features_2_bias, features_2_stride, features_2_padding, features_2_dilation, features_2_groups), 0), features_4_kernel_size, features_4_stride, features_4_padding), features_5_weight, features_5_bias, features_5_stride, features_5_padding, features_5_dilation, features_5_groups), 0), features_7_weight, features_7_bias, features_7_stride, features_7_padding, features_7_dilation, features_7_groups), 0), features_9_kernel_size, features_9_stride, features_9_padding), features_10_weight, features_10_bias, features_10_stride, features_10_padding, features_10_dilation, features_10_groups), 0), features_12_weight, features_12_bias, features_12_stride, features_12_padding, features_12_dilation, features_12_groups), 0), features_14_weight, features_14_bias, features_14_stride, features_14_padding, features_14_dilation, features_14_groups), 0), features_16_weight, features_16_bias, features_16_stride, features_16_padding, features_16_dilation, features_16_groups), 0), features_18_kernel_size, features_18_stride, features_18_padding), features_19_weight, features_19_bias, features_19_stride, features_19_padding, features_19_dilation, features_19_groups), 0), features_21_weight, features_21_bias, features_21_stride, features_21_padding, features_21_dilation, features_21_groups), 0), features_23_weight, features_23_bias, features_23_stride, features_23_padding, features_23_dilation, features_23_groups), 0), features_25_weight, features_25_bias, features_25_stride, features_25_padding, features_25_dilation, features_25_groups), 0), features_27_kernel_size, features_27_stride, features_27_padding), features_28_weight, features_28_bias, features_28_stride, features_28_padding, features_28_dilation, features_28_groups), 0), features_30_weight, features_30_bias, features_30_stride, features_30_padding, features_30_dilation, features_30_groups), 0), features_32_weight, features_32_bias, features_32_stride, features_32_padding, features_32_dilation, features_32_groups), 0), features_34_weight, features_34_bias, features_34_stride, features_34_padding, features_34_dilation, features_34_groups), 0), features_36_kernel_size, features_36_stride, features_36_padding)
    x = np.reshape(x, (x.shape[0], -1))
    x = ((np.maximum(((np.maximum(((x) @ classifier_0_weight.T + classifier_0_bias), 0)) @ classifier_3_weight.T + classifier_3_bias), 0)) @ classifier_6_weight.T + classifier_6_bias)
    return x

