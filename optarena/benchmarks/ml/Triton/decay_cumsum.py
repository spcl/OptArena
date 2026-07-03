import numpy as np

TRANSLATION_STATUS = 'partial'
TRANSLATION_UNSUPPORTED = {'launch_prepare_qg_kg': 'unsupported Triton wrapper pattern'}

def launch_fwd_decay_cumsum(g, g_o, B, H, T, scale, BT, BK, DK):
    return np.cumsum(g, axis=-1)

def launch_prepare_qg_kg(q, k, g, qg, kg, B, H, T, scale, BT, BK, DK):
    raise NotImplementedError('unsupported Triton wrapper pattern')

def launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, B, H, T, scale, BT, BK, DK):
    return np.cumsum(dq_inner, axis=-1)
