import numpy as np

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

class Model:
    def __init__(self, in_features, out_features, scaling_factor):
        self.matmul_weight = np.zeros((out_features, in_features), dtype=np.float32)
        self.matmul_bias = np.zeros((out_features,), dtype=np.float32) if True else np.zeros((out_features,), dtype=np.float32)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = ((x) @ self.matmul_weight.T + self.matmul_bias)
        x = (x * (1.0 / (1.0 + np.exp(-(x)))))
        x = (x * self.scaling_factor)
        return x

