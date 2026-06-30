# Copyright 2026 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""vexx_k C++ SoA numerical cross-check.

Drives the dace-fortran-generated C++ (``vexx_bp_k_core_generated.cpp``) of the
self-contained, FFT-free numeric core of ``exx_bp::vexx_bp_k`` -- the three
pointwise Fock stages (rhoc build -> vc scale -> result accumulate) lifted
verbatim from the inlined kernel (``vexx_bp_k_core.f90``) -- via ctypes, and
compares its output against the numpy reference (:func:`vexx_k_numpy._core`) on
the SAME random flat-SoA inputs.

WHY THIS SLICE: the full ``vexx_bp_k`` kernel does not lower through the
dace-fortran bridge end-to-end -- the two band-pair FFTs (``fwfft``/``invfft``)
are an irreducible external the inliner leaves unresolved (just as cegterg left
FFT / devxlib external), so flang rejects the empty generic interface at the call
sites.  So -- exactly like the sibling ``cegterg`` benchmark cross-checks only its
lowerable ``cegterg_rr`` Hermitianization core -- this harness cross-checks the
largest contiguous numeric stage of ``vexx_bp_k`` the bridge DOES lower cleanly to
compiling C++.  The full operator's correctness is covered by the property tests
in ``test_reference.py`` (Hermiticity + no-op + negrp invariance).

This inlining was produced with the SAME fparser pipeline used for ``cegterg``
(``dace_fortran.fparser_inliner.inline_to_ast(optimize=False)`` over the
preprocessed ``f2dace-qe-source`` tree), NOT the older f2dace AST-dump stack that
produced the legacy ``vexx`` benchmark's ``ast_v1_vexx_bp_k_gpu.f90``.

The DaCe runtime include dir is resolved via ``find_spec`` (no hardcoded paths);
the build-tree-relative ``#include "../../include/hash.h"`` (unused by this
kernel) is stripped before the standalone compile.  Skips cleanly when the
headers / a C++ compiler are unavailable.
"""
import ctypes
import importlib.util
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CPP = HERE / "vexx_bp_k_core_generated.cpp"
SO = HERE / "libvexx_bp_k_core.so"   # build artifact (gitignored; never committed)


def _dace_include():
    """Locate the DaCe runtime include dir without a hardcoded path: ``$DACE_DIR``,
    then the installed ``dace`` package (via ``find_spec``), then a sibling
    checkout.  Returns the dir or ``None`` (caller skips)."""
    def _ok(cand):
        return cand if (cand / "dace" / "dace.h").exists() else None

    env = os.environ.get("DACE_DIR")
    if env and _ok(pathlib.Path(env) / "dace" / "runtime" / "include"):
        return pathlib.Path(env) / "dace" / "runtime" / "include"
    spec = importlib.util.find_spec("dace")
    if spec is not None:
        locs = list(spec.submodule_search_locations or [])
        if not locs and spec.origin:
            locs = [str(pathlib.Path(spec.origin).parent)]
        for loc in locs:
            hit = _ok(pathlib.Path(loc) / "runtime" / "include")
            if hit:
                return hit
    for anc in HERE.parents:
        hit = _ok(anc / "dace" / "dace" / "runtime" / "include")
        if hit:
            return hit
    return None


def _ensure_so():
    """Build ``libvexx_bp_k_core.so`` from the generated C++ if absent.  Strips the
    build-tree-relative ``hash.h`` include (unused here).  Returns the .so path,
    or ``None`` when the DaCe headers / a C++ compiler aren't available."""
    if SO.exists():
        return SO
    inc = _dace_include()
    if inc is None:
        return None
    src = CPP.read_text()
    src = "\n".join(ln for ln in src.splitlines() if "include/hash.h" not in ln)
    patched = HERE / "_vexx_bp_k_core_standalone.cpp"
    patched.write_text(src)
    r = subprocess.run(["g++", "-O2", "-std=c++17", "-fPIC", "-shared",
                        f"-I{inc}", str(patched), "-o", str(SO)],
                       capture_output=True, text=True)
    return SO if r.returncode == 0 else None


def run_cpp(exxbuff, facb, temppsic, result, occ, omega_inv, nqs_inv):
    """Run the generated C++ Fock core on F-ordered SoA inputs; returns ``result``
    (``(nrxxs,)`` complex128) accumulated in place.

    ``exxbuff``: ``(nrxxs, jcount)`` complex128, ``facb``: ``(nrxxs,)`` float64,
    ``temppsic`` / ``result``: ``(nrxxs,)`` complex128."""
    nrxxs, jcount = exxbuff.shape
    exxbuff = np.asfortranarray(exxbuff.astype(np.complex128))
    facb = np.ascontiguousarray(facb.astype(np.float64))
    temppsic = np.ascontiguousarray(temppsic.astype(np.complex128))
    result = np.ascontiguousarray(result.astype(np.complex128))
    lib = ctypes.CDLL(str(SO))
    lib.__dace_init_vexx_bp_k_core.restype = ctypes.c_void_p
    lib.__dace_init_vexx_bp_k_core.argtypes = [ctypes.c_int, ctypes.c_int]
    state = lib.__dace_init_vexx_bp_k_core(int(jcount), int(nrxxs))
    # __program signature: (state, exxbuff, facb, result, temppsic, jcount,
    #                       nqs_inv, nrxxs, occ, omega_inv)
    lib.__program_vexx_bp_k_core.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_double]
    lib.__program_vexx_bp_k_core(
        ctypes.c_void_p(state),
        exxbuff.ctypes.data_as(ctypes.c_void_p),
        facb.ctypes.data_as(ctypes.c_void_p),
        result.ctypes.data_as(ctypes.c_void_p),
        temppsic.ctypes.data_as(ctypes.c_void_p),
        int(jcount), float(nqs_inv), int(nrxxs), float(occ), float(omega_inv))
    lib.__dace_exit_vexx_bp_k_core.argtypes = [ctypes.c_void_p]
    lib.__dace_exit_vexx_bp_k_core(ctypes.c_void_p(state))
    return result


def main():
    if _ensure_so() is None:
        print("SKIP: DaCe runtime headers / g++ not found -- cannot build the C++ "
              "SoA reference; skipping cross-check.")
        return
    spec = importlib.util.spec_from_file_location("vexx_k_numpy", HERE.parent / "vexx_k_numpy.py")
    knp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(knp)

    rng = np.random.default_rng(0)
    nrxxs, jcount = 200, 5
    occ, omega_inv, nqs_inv = 1.7, 1.0 / 1.3, 1.0
    exxbuff = (rng.standard_normal((nrxxs, jcount)) + 1j * rng.standard_normal((nrxxs, jcount)))
    facb = rng.standard_normal(nrxxs)
    temppsic = (rng.standard_normal(nrxxs) + 1j * rng.standard_normal(nrxxs))
    result0 = (rng.standard_normal(nrxxs) + 1j * rng.standard_normal(nrxxs))

    res_np = knp._core(exxbuff.copy(), facb.copy(), temppsic.copy(), result0.copy(),
                       occ, omega_inv, nqs_inv)
    res_cpp = run_cpp(exxbuff.copy(), facb.copy(), temppsic.copy(), result0.copy(),
                      occ, omega_inv, nqs_inv)
    d = float(np.abs(res_np - res_cpp).max())
    print(f"max|result_np - result_cpp| = {d:.3e}")
    print("MATCH" if d < 1e-12 else "MISMATCH")


if __name__ == "__main__":
    main()
