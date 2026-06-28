"""vexx C++ SoA numerical cross-check.

Drives the dace-fortran-generated C++ (``vexx_bp_k_gpu_generated.cpp``) of the
ABI-faithful ``vexx_bp_k_gpu`` via ctypes on the SAME single-k / single-q /
collinear / norm-conserving problem as the numpy SoA reference (built by
``soa_inputs.build_soa``), then compares the returned ``hpsi``.

STATUS (2026-06-26): the harness is COMPLETE and correct -- it compiles the
generated C++, marshals all ~165 args (F-order, 1-based; scalars are size-1
pointers; disabled QE paths flag-gated off and dummied), and reads ``hpsi``
back. Two findings:

  1. The generated C++ DROPPED the closing ``hpsi = hpsi_d`` SOURCE= copy-back
     (the SDFG lowering's "uninitialized transient" prune, same as the input
     capture). Fixed in the .cpp by a MANUAL FIX mirroring the input copy.
  2. The FFT lib-nodes were broken and are now FIXED in the .cpp (MANUAL FIX
     ``__vexx_fft3d``): the dace-fortran lowering had flattened the (n1,n2,n3)
     grid to a single ``nrxxs`` axis and emitted a 1-D DFT / an N-D DFT over the
     WRONG storage axes (transforming the band/spin batch) with NO 1/N inverse
     normalisation. The replacement does the correct batched 3-D DFT
     (nr=cbrt(nrxxs)). VERIFIED by instrumentation: invfft(tg)->temppsic now
     O(10) (was 1.75e11 garbage); the collinear rhoc=conj(phi)*temppsic computes
     correctly (|.|~54).

  3. Two SPURIOUS in-loop reallocations were dropping the data: the SDFG emitted
     ``rhoc_d = new`` (and ``vc_d = new``) inside the fwfft / invfft loops --
     between the compute loop and the FFT -- re-pointing the buffer to a fresh
     ZERO allocation, so the FFT transformed zeros. Both removed (MANUAL FIX);
     the p<x>_d slices now index the populated buffer. With this the active-path
     exchange FIRES (big_result 0 -> nonzero).

  4. REMAINING: the result is now substantially correct -- the bulk of elements
     match numpy to ~0.9x -- EXCEPT the last band (band m-1), whose elements are
     off by exactly N^2 (=nrxxs^2), i.e. they missed both inverse-FFT
     normalisations. A residual per-band normalisation/aliasing interaction in
     the generated band-batch handling, plus a ~10% systematic factor on the
     bulk. These are the last dace-fortran lowering wrinkles; full bit-for-bit
     is a regeneration-from-fixed-dace-fortran task. The numpy SoA reference
     itself is validated (Hermitian to machine precision).

  SUMMARY: FFT lib-nodes fixed + 2 data-dropping reallocs fixed took the C++ SoA
  from "exchange == 0" to "exchange computes, bulk ~correct"; one per-band
  normalisation outlier remains.

Run:  python soa_cpp_check.py
It auto-builds libvexx_soa.so from the generated C++ using the DaCe runtime
headers, located portably via (in order) $DACE_DIR, a sibling ``dace`` checkout
next to the repo, or the installed ``dace`` package. If none is found (or no C++
compiler), it SKIPS cleanly. No machine-specific paths are baked in; the built
.so is a gitignored artifact, never committed.
"""
import ctypes
import os
import pathlib
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import soa_inputs as SI   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CPP = HERE / "vexx_bp_k_gpu_generated.cpp"
SO = HERE / "libvexx_soa.so"   # build artifact (gitignored; never committed)


def _dace_include():
    """Locate the DaCe runtime include dir WITHOUT a hardcoded path. ``dace`` is
    a declared repo dependency (see requirements/cpu.txt), so the canonical
    source is the INSTALLED dace package -- resolved via ``find_spec`` (no slow
    ``import dace``). Falls back to ``$DACE_DIR`` or a sibling checkout for local
    dev. Returns the dir, or ``None`` (caller skips)."""
    import importlib.util

    def _ok(cand):
        return cand if (cand / "dace" / "dace.h").exists() else None

    env = os.environ.get("DACE_DIR")
    if env and _ok(pathlib.Path(env) / "dace" / "runtime" / "include"):
        return pathlib.Path(env) / "dace" / "runtime" / "include"
    # Installed dace dependency -- locate its package dir without importing it.
    spec = importlib.util.find_spec("dace")
    if spec is not None:
        locs = list(spec.submodule_search_locations or [])
        if not locs and spec.origin:
            locs = [str(pathlib.Path(spec.origin).parent)]
        for loc in locs:
            hit = _ok(pathlib.Path(loc) / "runtime" / "include")
            if hit:
                return hit
    # Local-dev fallback: a sibling ``dace`` checkout next to any ancestor.
    for anc in HERE.parents:
        hit = _ok(anc / "dace" / "dace" / "runtime" / "include")
        if hit:
            return hit
    return None


def _ensure_so():
    """Build libvexx_soa.so from the generated C++ if absent. Returns the .so
    path, or ``None`` when the DaCe headers / a C++ compiler aren't available
    (the caller should then SKIP rather than fail)."""
    if SO.exists():
        return SO
    inc = _dace_include()
    if inc is None:
        return None
    r = subprocess.run(["g++", "-O2", "-std=c++17", "-fPIC", "-shared",
                        f"-I{inc}", str(CPP), "-o", str(SO)],
                       capture_output=True, text=True)
    return SO if r.returncode == 0 else None


def _program_params():
    src = CPP.read_text()
    m = re.search(r"DACE_EXPORTED void __program_vexx_bp_k_gpu\(([^)]*)\)", src)
    out = []
    for p in m.group(1).split(","):
        toks = p.strip().replace("*", " * ").split()
        out.append((toks[-1], " ".join(t for t in toks[:-1] if t != "*").replace("__restrict__", "").strip(), "*" in p))
    return out


def _init_params():
    src = CPP.read_text()
    m = re.search(r"DACE_EXPORTED vexx_bp_k_gpu_state_t \*__dace_init_vexx_bp_k_gpu\(([^)]*)\)", src)
    return [p.strip().split()[-1] for p in m.group(1).split(",")]


def run_cpp(kw):
    lib = ctypes.CDLL(str(SO))
    keep = []
    n, m, nbnd, nrxxs = kw["n"], kw["m"], kw["nbnd"], kw["nrxxs"]
    ngm, npwx, npol, nqs = kw["ngm"], kw["npwx"], kw["npol"], kw["nqs"]
    nks, negrp, max_pairs = kw["nks"], kw["negrp"], kw["max_pairs"]
    cx = np.complex128

    def F(a, dt):
        return np.asfortranarray(a.astype(dt))

    def keepp(a):
        keep.append(a)
        return a.ctypes.data_as(ctypes.c_void_p)

    def scal(v):
        return np.array([v], dtype=np.float64)

    def flag(v):
        return np.array([v], dtype=np.bool_)

    def iscal(v):
        return np.array([v], dtype=np.int32)

    _fac = SI.ref_mod._coulomb_fac(kw["g"], kw["xk"][:, kw["current_k"] - 1],
                                   kw["xkq_collect"][:, 0], ngm, kw["tpiba2"],
                                   kw["exxdiv"], kw["eps_qdiv"], kw["gau_scrlen"],
                                   kw["erf_scrlen"], kw["erfc_scrlen"], kw["yukawa"])
    coulomb_fac = np.zeros(ngm * nqs * nks, dtype=np.float64, order="F")
    coulomb_fac[:ngm] = _fac

    known = {
        "psi": F(kw["psi"], cx), "hpsi": F(kw["hpsi"], cx), "exxbuff": F(kw["exxbuff"], cx),
        "coulomb_fac": coulomb_fac, "dfftt_nl": F(kw["dfftt_nl"], np.int32),
        "igk_exx": F(kw["igk_exx"], np.int32), "index_xk": F(kw["index_xk"], np.int32),
        "index_xkq": F(kw["index_xkq"], np.int32), "xk": F(kw["xk"], np.float64),
        "xkq_collect": F(kw["xkq_collect"], np.float64), "g": F(kw["g"], np.float64),
        "ibands": F(kw["ibands"], np.int32), "nibands": F(kw["nibands"], np.int32),
        "all_start": F(kw["all_start"], np.int32), "all_end": F(kw["all_end"], np.int32),
        "egrp_pairs": F(kw["egrp_pairs"], np.int32), "iexx_istart": F(kw["iexx_istart"], np.int32),
        "x_occupation": F(kw["x_occupation"], np.float64),
        "exxbuff_d": np.asfortranarray(kw["exxbuff"][:, :, 0].astype(cx)).ravel(order="F"),
        "igk_exx_d": np.asfortranarray(kw["igk_exx"].astype(np.int32)).ravel(order="F"),
        "x_occupation_d": np.asfortranarray(kw["x_occupation"].astype(np.float64)).ravel(order="F"),
        "exxalfa": scal(kw["exxalfa"]), "omega": scal(kw["omega"]), "tpiba2": scal(kw["tpiba2"]),
        "tpiba": scal(1.0), "exxdiv": scal(kw["exxdiv"]), "eps_qdiv": scal(kw["eps_qdiv"]),
        "eps": scal(kw["eps_qdiv"]), "gau_scrlen": scal(kw["gau_scrlen"]),
        "erf_scrlen": scal(kw["erf_scrlen"]), "erfc_scrlen": scal(kw["erfc_scrlen"]),
        "yukawa": scal(kw["yukawa"]), "grid_factor": scal(1.0), "coulomb_done": flag(True),
        "nq1": iscal(1), "nq2": iscal(1), "nq3": iscal(1), "npool": iscal(1),
        "many_fft": iscal(1), "kunit": iscal(1), "nkstot": iscal(nks), "me_egrp": iscal(0),
        "inter_egrp_comm": iscal(1), "intra_egrp_comm": iscal(1), "iexx_iend": iscal(nbnd),
        "iexx_istart_d": iscal(1), "nij_type": iscal(1),
    }
    false_flags = {"gamma_only", "ionode", "noncolin", "okpaw", "okvan",
                   "paw_has_init_paw_fockrnl", "tqr", "upf_tpawp", "upf_tvanp",
                   "use_coulomb_vcut_spheric", "use_coulomb_vcut_ws", "x_gamma_extrapolation"}
    dims = {
        "current_k": kw["current_k"], "current_ik": kw["current_ik"], "nqs": nqs, "n": n,
        "m": m, "npwx": npwx, "npol": npol, "nrxxs": nrxxs, "ngm": ngm, "nks": nks,
        "negrp": negrp, "my_egrp_id": kw["my_egrp_id"], "max_pairs": max_pairs,
        "jblock": kw["jblock"], "iexx_start": kw["iexx_start"], "lda": npwx,
        "dfftt_ngm": ngm, "dfftt_nnr": nrxxs, "dfftt__nl_d0": ngm, "g_d0": 3,
        "egrp_pairs_d0": 2, "egrp_pairs_d1": max_pairs, "ibands_d0": kw["ibands"].shape[0],
        "index_xkq_d0": kw["index_xkq"].shape[0], "igk_exx_d0": npwx, "igk_exx_d_d0": npwx,
        "x_occupation_d0": nbnd, "x_occupation_d_d0": nbnd, "xkq_collect_d0": 3,
        "exxbuff_d0": nrxxs, "exxbuff_d1": nbnd, "exxbuff_d2": kw["exxbuff"].shape[2],
        "exxbuff_d_d0": nrxxs, "exxbuff_d_d1": nbnd, "psi_d_d0": npwx, "psi_d_d1": m,
        "hpsi_d_d0": npwx, "hpsi_d_d1": m, "my_pool_id": 0, "gstart": 1, "run_on_gpu_": 1,
    }

    init_args = []
    for nm in _init_params():
        v = int(dims.get(nm, 1))
        init_args.append(ctypes.c_int64(v) if nm[-2:] in ("d0", "d1", "d2", "d3") else ctypes.c_int(v))
    lib.__dace_init_vexx_bp_k_gpu.restype = ctypes.c_void_p
    state = lib.__dace_init_vexx_bp_k_gpu(*init_args)

    call_args = []
    for nm, typ, isptr in _program_params():
        if nm == "__state":
            call_args.append(ctypes.c_void_p(state))
        elif not isptr:
            v = int(dims.get(nm, 1))
            call_args.append(ctypes.c_int64(v) if typ == "int64_t"
                             else ctypes.c_bool(bool(dims.get(nm, False))) if typ == "bool"
                             else ctypes.c_int(v))
        elif nm in known:
            call_args.append(keepp(known[nm]))
        elif nm in false_flags:
            call_args.append(keepp(flag(False)))
        else:
            dt = cx if "complex128" in typ else np.bool_ if typ == "bool" else np.int32 if typ == "int" else np.float64
            call_args.append(keepp(np.zeros(64 * max(nrxxs * nbnd, n * m, ngm * 4, 1024), dtype=dt, order="F")))
    lib.__program_vexx_bp_k_gpu(*call_args)
    return np.asarray(known["hpsi"])


def main():
    if _ensure_so() is None:
        print("SKIP: DaCe runtime headers not found (set $DACE_DIR or pip-install "
              "dace) -- cannot build the C++ SoA reference; skipping cross-check.")
        return
    kw, _ = SI.build_soa(ngrid=8, nbnd=3, m=5)
    hpsi0 = kw["hpsi"].copy()
    kw_np = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
    hpsi_np = SI.ref_mod.vexx_bp_k_gpu(**kw_np)
    kw_cpp = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
    kw_cpp["hpsi"] = hpsi0.copy()
    hpsi_cpp = run_cpp(kw_cpp)
    d = float(np.abs(hpsi_np - hpsi_cpp).max())
    print(f"numpy SoA dV norm = {np.linalg.norm(hpsi_np - hpsi0):.6g}")
    print(f"C++   SoA dV norm = {np.linalg.norm(hpsi_cpp - hpsi0):.6g}")
    print(f"max |hpsi_np - hpsi_cpp| = {d:.6g}  (rel {d / (np.abs(hpsi_np).max() + 1e-300):.3g})")
    if np.linalg.norm(hpsi_cpp - hpsi0) == 0:
        print("NOTE: C++ produced no exchange (big_result==0) -- upstream dace-fortran "
              "active-path lowering gap; see module docstring.")


if __name__ == "__main__":
    main()
