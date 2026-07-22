from scipy.linalg import eigh as _sci_eigh
import numpy as np


def eigh_test(a, b, wout, vout):
    w, v = _sci_eigh(a, b, lower=False)
    wout[:] = w
    vout[:] = v
