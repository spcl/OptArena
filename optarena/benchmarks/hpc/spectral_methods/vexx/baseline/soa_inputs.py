"""vexx SoA cross-check harness: build the ~50 SoA inputs for
``vexx_bp_k_gpu`` so the active path reduces to the compact ``vexx`` problem,
then (step 1) run the numpy SoA reference and check it is Hermitian + agrees
with compact vexx when the coulomb factor is made consistent.

Single k-point, single q (Gamma), collinear, norm-conserving, negrp=1, occ=1.
"""
import importlib.util
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent   # the vexx benchmark dir


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


vexx_mod = _load("vexx", "vexx.py")
ref_mod = _load("vexx_numpy", "vexx_numpy.py")


def build_soa(ngrid=8, nbnd=3, m=5):
    """Build SoA inputs whose active path == the compact vexx problem.

    Builds the collinear / norm-conserving compact problem INLINE (the benchmark
    ``initialize`` now takes config flags and returns the full all-paths tuple, so
    this cross-check -- pinned to the single-k / single-q / NC config the
    generated C++ supports -- constructs its own minimal inputs).

    Returns (kwargs_for_soa, compact_inputs, mill)."""
    from numpy.random import default_rng
    rng = default_rng(0)
    n1 = n2 = n3 = ngrid
    nnr = n1 * n2 * n3
    grid = (n1, n2, n3)
    hmax0 = ngrid // 2 - 1
    cut0 = hmax0 ** 2
    nl_list, g2 = [], []
    rh = range(-hmax0, hmax0 + 1)
    for hx in rh:
        for hy in rh:
            for hz in rh:
                if hx * hx + hy * hy + hz * hz <= cut0:
                    nl_list.append(np.ravel_multi_index((hx % n1, hy % n2, hz % n3), grid))
                    g2.append(hx * hx + hy * hy + hz * hz)
    nl_c = np.array(nl_list, dtype=np.int32)
    npw = len(nl_c)
    g2 = np.array(g2, dtype=np.float64)
    coulomb_fac_c = np.where(g2 > 0, 1.0 / np.where(g2 > 0, g2, 1.0), 0.0)
    psi = (rng.standard_normal((npw, m)) + 1j * rng.standard_normal((npw, m)))
    hpsi = (rng.standard_normal((npw, m)) + 1j * rng.standard_normal((npw, m)))
    exxbuff_c = (rng.standard_normal((nnr, nbnd)) + 1j * rng.standard_normal((nnr, nbnd)))
    occ = np.ones(nbnd, dtype=np.float64)
    exxalfa, omega, nqs = 0.25, 1.0, 1
    n = npw
    ngm = npw
    nrxxs = nnr

    # 1-based FFT-grid index table; gki = identity so nlg == nl_c.
    dfftt_nl = (nl_c.astype(np.int64) + 1)                # (ngm,)
    igk_exx = np.arange(1, n + 1, dtype=np.int64).reshape(n, 1)   # (npwx, nks)

    index_xkq = np.array([[1]], dtype=np.int64)           # (nks_ik, nqs) -> ikq=1
    index_xk = np.array([1], dtype=np.int64)              # (nkq,) -> ik=1
    xk = np.zeros((3, 1))                                 # single k at Gamma
    xkq_collect = np.zeros((3, 1))                        # q-shift = 0 -> q = g

    # G-vectors: recover miller indices from the same construction as initialize.
    hmax = ngrid // 2 - 1
    cutoff2 = hmax ** 2
    mill = []
    rng_h = range(-hmax, hmax + 1)
    for hx in rng_h:
        for hy in rng_h:
            for hz in rng_h:
                if hx * hx + hy * hy + hz * hz <= cutoff2:
                    mill.append((hx, hy, hz))
    mill = np.array(mill, dtype=np.float64)               # (npw, 3)
    assert mill.shape[0] == npw
    g = np.zeros((3, ngm))
    g[:, :ngm] = mill.T

    # exxbuff: (nrxxs, nbnd, nkq) -- compact (nnr, nbnd) on the single kq.
    exxbuff = exxbuff_c[:, :, None].copy()

    # x_occupation: (nbnd, nks) all 1.
    x_occupation = np.ones((nbnd, 1))

    # ibands: my bands = the m trial bands (1..m); nibands[eg] = m.
    ibands = np.arange(1, m + 1, dtype=np.int64).reshape(m, 1)   # (my_n, negrp)
    nibands = np.array([m], dtype=np.int64)

    all_start = np.array([1], dtype=np.int64)
    all_end = np.array([nbnd], dtype=np.int64)

    # egrp_pairs: pair every trial band ibnd (1..m) with every occupied j (1..nbnd).
    # shape (2, max_pairs, negrp); [0]=trial band, [1]=occupied orbital.
    pairs = [(ib, j) for ib in range(1, m + 1) for j in range(1, nbnd + 1)]
    max_pairs = len(pairs)
    egrp_pairs = np.zeros((2, max_pairs, 1), dtype=np.int64)
    for ip, (ib, j) in enumerate(pairs):
        egrp_pairs[0, ip, 0] = ib
        egrp_pairs[1, ip, 0] = j
    iexx_istart = np.array([1], dtype=np.int64)           # istart > 0 -> finalize fires

    kw = dict(
        psi=psi, hpsi=hpsi, exxbuff=exxbuff, x_occupation=x_occupation,
        coulomb_fac=coulomb_fac_c, dfftt_nl=dfftt_nl, igk_exx=igk_exx,
        index_xk=index_xk, index_xkq=index_xkq, xk=xk, xkq_collect=xkq_collect,
        g=g, ibands=ibands, nibands=nibands, all_start=all_start, all_end=all_end,
        egrp_pairs=egrp_pairs, iexx_istart=iexx_istart, exxalfa=exxalfa, omega=omega,
        tpiba2=1.0, exxdiv=0.0, eps_qdiv=1e-8, gau_scrlen=0.0, erf_scrlen=0.0,
        erfc_scrlen=0.0, yukawa=0.0, current_k=1, current_ik=1, nqs=1, n=n, m=m,
        npwx=n, npol=1, nrxxs=nrxxs, ngm=ngm, nks=1, n1=n1, n2=n2, n3=n3,
        nbnd=nbnd, my_egrp_id=0, max_pairs=max_pairs, jblock=nbnd, negrp=1,
        iexx_start=1)
    compact = dict(psi=psi, exxbuff=exxbuff_c, occ=occ, coulomb_fac=coulomb_fac_c,
                   nl=nl_c, exxalfa=exxalfa, omega=omega, nqs=nqs, npw=npw, m=m,
                   nbnd=nbnd, nnr=nnr, n1=n1, n2=n2, n3=n3, mill=mill)
    return kw, compact


if __name__ == "__main__":
    kw, compact = build_soa()
    hpsi0 = kw["hpsi"].copy()
    # numpy SoA reference run
    out = ref_mod.vexx(**kw)
    dV = out - hpsi0
    print("SoA: ||hpsi change|| =", float(np.linalg.norm(dV)))
    print("SoA: any nan?", bool(np.isnan(dV).any()))

    # Hermiticity of the SoA Vx (apply to zero accumulator).
    kw2, _ = build_soa()
    kw2["hpsi"] = np.zeros_like(kw2["psi"])
    Vx = ref_mod.vexx(**kw2)
    mtx = kw2["psi"].conj().T @ Vx
    herm = np.abs(mtx - mtx.conj().T).max() / (np.abs(mtx).max() + 1e-300)
    print("SoA: ||Vx|| =", float(np.linalg.norm(Vx)), " Hermiticity =", herm)
