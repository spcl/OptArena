# Copyright 2026 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""cegterg C++ SoA numerical cross-check.

Drives the dace-fortran-generated C++ (``cegterg_rr_generated.cpp``) of the
self-contained Rayleigh-Ritz Hermitianization core (cegterg.f90:730-737 -- the
"diagonal of hc and sc must be strictly real" + conjugate-symmetry step) via
ctypes, and compares its output against the numpy reference
(:func:`cegterg_numpy._hermitianize`) on the SAME random reduced ``hc`` / ``sc``.

WHY THIS SLICE: the full ``cegterg`` kernel does not yet lower through the
dace-fortran bridge end-to-end -- the SDFG build hits genuine upstream gaps on
this kernel (an allocatable-conditional operand the C++ AST builder cannot trace,
plus unhandled HLFIR ops ``hlfir.count`` for ``COUNT(.NOT.conv)``, ``fir.negc``
for the complex ``-ew*psi`` negation, and ``scf.index_switch``).  So -- exactly
as the sibling ``vexx`` benchmark cross-checks only its lowerable collinear-NC
SoA slice -- this harness cross-checks the largest self-contained numeric stage
of cegterg that the bridge DOES lower cleanly to compiling C++.  The full
solver's correctness is covered by the property tests in ``test_reference.py``
(converged eigenvalues == direct generalised ``eigh``).

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
CPP = HERE / "cegterg_rr_generated.cpp"
SO = HERE / "libcegterg_rr.so"   # build artifact (gitignored; never committed)


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
    """Build ``libcegterg_rr.so`` from the generated C++ if absent.  Strips the
    build-tree-relative ``hash.h`` include (unused here).  Returns the .so path,
    or ``None`` when the DaCe headers / a C++ compiler aren't available."""
    if SO.exists():
        return SO
    inc = _dace_include()
    if inc is None:
        return None
    src = CPP.read_text()
    src = "\n".join(ln for ln in src.splitlines() if "include/hash.h" not in ln)
    patched = HERE / "_cegterg_rr_standalone.cpp"
    patched.write_text(src)
    r = subprocess.run(["g++", "-O2", "-std=c++17", "-fPIC", "-shared",
                        f"-I{inc}", str(patched), "-o", str(SO)],
                       capture_output=True, text=True)
    return SO if r.returncode == 0 else None


def run_cpp(hc, sc, nbase):
    """Run the generated C++ Hermitianization on F-ordered ``hc`` / ``sc``
    (``(nvecx, nvecx)`` complex128), in place; returns ``(hc, sc)``."""
    nvecx = hc.shape[0]
    hc = np.asfortranarray(hc.astype(np.complex128))
    sc = np.asfortranarray(sc.astype(np.complex128))
    lib = ctypes.CDLL(str(SO))
    lib.__dace_init_cegterg_rr.restype = ctypes.c_void_p
    lib.__dace_init_cegterg_rr.argtypes = [ctypes.c_int, ctypes.c_int]
    state = lib.__dace_init_cegterg_rr(int(nbase), int(nvecx))
    lib.__program_cegterg_rr.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.__program_cegterg_rr(ctypes.c_void_p(state),
                             hc.ctypes.data_as(ctypes.c_void_p),
                             sc.ctypes.data_as(ctypes.c_void_p),
                             int(nbase), int(nvecx))
    lib.__dace_exit_cegterg_rr.argtypes = [ctypes.c_void_p]
    lib.__dace_exit_cegterg_rr(ctypes.c_void_p(state))
    return hc, sc


def main():
    if _ensure_so() is None:
        print("SKIP: DaCe runtime headers / g++ not found -- cannot build the C++ "
              "SoA reference; skipping cross-check.")
        return
    spec = importlib.util.spec_from_file_location("cegterg_numpy", HERE.parent / "cegterg_numpy.py")
    knp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(knp)

    rng = np.random.default_rng(0)
    nvecx, nbase = 12, 8
    hc = (rng.standard_normal((nvecx, nvecx)) + 1j * rng.standard_normal((nvecx, nvecx)))
    sc = (rng.standard_normal((nvecx, nvecx)) + 1j * rng.standard_normal((nvecx, nvecx)))
    hc_np, sc_np = hc.copy(), sc.copy()
    knp._hermitianize(hc_np, sc_np, nbase)
    hc_cpp, sc_cpp = run_cpp(hc.copy(), sc.copy(), nbase)
    dh = float(np.abs(hc_np - hc_cpp).max())
    ds = float(np.abs(sc_np - sc_cpp).max())
    print(f"max|hc_np - hc_cpp| = {dh:.3e}")
    print(f"max|sc_np - sc_cpp| = {ds:.3e}")
    print("MATCH" if max(dh, ds) < 1e-12 else "MISMATCH")


if __name__ == "__main__":
    main()
