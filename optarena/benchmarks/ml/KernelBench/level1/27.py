import numpy as np

batch_size = 4096
dim = 393216

class Model:
    def __init__(self):
        pass

    def forward(self, x):
        return (1.0507009873554805 * np.where((x) > 0, (x), 1.6732632423543772 * (np.exp(x) - 1.0)))

