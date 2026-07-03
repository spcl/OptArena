import numpy as np

TRANSLATION_STATUS = 'translated'
TRANSLATION_UNSUPPORTED = {}

def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    return np.array(old_a_start, copy=True)
