import numpy as np

batch_size = 4096
dim = 393216

class Model:
    def __init__(self):
        pass

    def forward(self, x):
        return (1.0 / (1.0 + np.exp(-(x))))

