import numpy as np

b = 8
i = 256
j = 512
l = 256
k = 768

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        return np.einsum('bijl,lk->bijk', A, B)

