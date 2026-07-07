# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The distributed build + runner (``sandbox.build_mpi``, ``languages.build_mpi_executable_commands``,
``agent_bench/mpi_call.run``).

``mpi_call.run`` is the 5th runner: it must return the SAME ``(outputs, native_ns)`` shape as the
single-node ``_call_isolated`` so the grading + metric core is reused verbatim. These tests pin
the build command shape (pure, no compiler) and -- gated on a working MPI toolchain -- the full
build -> scatter -> launch -> gather round-trip, asserting the reconstructed global output equals
the reference and that a launcher failure is a SCORED ``RuntimeError``.
"""
from pathlib import Path

import numpy as np
import pytest

from optarena.agent_bench import mpi_call
from optarena.agent_bench.envelope import Submission
from optarena.agent_bench.mpi_descriptor import ArrayDist, AxisDist, Descriptor, Grid
from optarena.agent_bench.sandbox import Sandbox
from optarena.agent_bench.task import Task
from optarena.bindings.contract import Arg, Binding
from optarena.bindings.stubs import LANGS
from optarena.languages import build_mpi_executable_commands
from tests.mpi_launch_helpers import c_toolchain, cc_override_for

RANKS = 4
_C_KERNEL = """
#include <mpi.h>
#include <stdint.h>
void yax_mpi(const double *restrict x, double *restrict y, const int64_t N, const double a,
             MPI_Fint comm, uint8_t *restrict workspace, const int64_t workspace_size) {
    for (int64_t i = 0; i < N; i++) y[i] = a * x[i];
}
"""


def _yax_binding() -> Binding:
    args = (
        Arg(name="x", kind="ptr", dtype="float64", is_const=True),
        Arg(name="y", kind="ptr", dtype="float64", is_const=False, role="output"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
        Arg(name="a", kind="scalar", dtype="float64", is_const=True),
    )
    return Binding(kernel="yax", config="dense", args=args, symbols={lang: "yax_fp64" for lang in LANGS})


def _descriptor() -> Descriptor:
    block0 = ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), ))
    return Descriptor(grid=Grid((RANKS, )), arrays={"x": block0, "y": block0}, symbol_axes={"N": [("x", 0)]})


# --------------------------------------------------------------------------------------- #
# build_mpi_executable_commands -- pure command shape (no compiler needed)
# --------------------------------------------------------------------------------------- #
def test_build_commands_compile_each_source_and_link_executable():
    cmds = build_mpi_executable_commands([("c", Path("k.c"))], Path("d.c"), Path("bench"))
    assert len(cmds) == 3  # compile kernel, compile driver, link
    # the MPICH C wrapper is the default; each source compiles to an object.
    assert cmds[0][0] == "mpicc.mpich" and "-c" in cmds[0] and str(Path("k.c")) in cmds[0]
    assert cmds[1][0] == "mpicc.mpich" and str(Path("d.c")) in cmds[1]
    # the link produces an EXECUTABLE (never -shared) and names the exe.
    link = cmds[-1]
    assert "-shared" not in " ".join(link)
    assert "-o" in link and str(Path("bench")) in link


def test_build_commands_cc_override_swaps_wrapper():
    cmds = build_mpi_executable_commands([("c", Path("k.c"))], Path("d.c"), Path("bench"), cc_override={"c": "mpicc"})
    assert all(argv[0] == "mpicc" for argv in cmds)  # OpenMPI wrapper for a matching launcher


def test_build_commands_fortran_kernel_links_with_fortran_driver():
    # A Fortran kernel + the always-C driver -> link with the Fortran wrapper (pulls libgfortran).
    cmds = build_mpi_executable_commands([("fortran", Path("k.f90"))], Path("d.c"), Path("bench"))
    assert cmds[0][0] == "mpifort.mpich"  # kernel compiled with the Fortran wrapper
    assert cmds[1][0] == "mpicc.mpich"  # driver always C
    assert cmds[-1][0] == "mpifort.mpich"  # link driver = Fortran


def test_build_commands_baseline_flows_from_matrix_no_literal_flags():
    cmds = build_mpi_executable_commands([("c", Path("k.c"))], Path("d.c"), Path("bench"))
    # -O3/-march come from the matrix baseline, never hard-coded in the yaml template.
    assert any("-O3" in tok for tok in cmds[0])


def test_build_commands_empty_sources_raises():
    with pytest.raises(ValueError, match="no kernel sources"):
        build_mpi_executable_commands([], Path("d.c"), Path("bench"))


# --------------------------------------------------------------------------------------- #
# Sandbox.build_mpi -- delivery handling
# --------------------------------------------------------------------------------------- #
def test_build_mpi_any_delivery_unsupported():
    b = _yax_binding()
    sub = Submission(language="c", library="/tmp/does-not-matter.so")
    with Sandbox(Task(kernel="yax"), b) as sb:
        res = sb.build_mpi(sub, _descriptor())
    assert not res.ok and "not supported" in res.log


def test_build_mpi_python_delivery_stashes_module():
    b = _yax_binding()
    sub = Submission(language="python", source="def kernel_mpi(*a, **k): pass\n")
    with Sandbox(Task(kernel="yax"), b) as sb:
        res = sb.build_mpi(sub, _descriptor())
        assert res.ok and res.exe is None and res.lib is not None
        assert res.lib.read_text().startswith("def kernel_mpi")


# --------------------------------------------------------------------------------------- #
# End to end: build -> scatter -> launch -> gather (gated on a working MPI toolchain)
# --------------------------------------------------------------------------------------- #
def test_build_mpi_and_run_round_trip(tmp_path):
    tc = c_toolchain()
    if tc is None:
        pytest.skip("no working MPI C compiler + launcher in this environment")
    cc, launch = tc
    b, desc = _yax_binding(), _descriptor()
    sub = Submission(language="c", source=_C_KERNEL)
    N = 20
    x = np.arange(N, dtype=np.float64) + 1.0
    data = {"x": x, "y": np.zeros(N), "N": N, "a": 3.0}

    with Sandbox(Task(kernel="yax"), b) as sb:
        built = sb.build_mpi(sub, desc, cc_override=cc_override_for(cc))
        assert built.ok, built.log
        assert built.exe is not None and built.exe.exists()
        outputs, native_ns = mpi_call.run(built.exe,
                                          b,
                                          desc,
                                          data,
                                          is_python=False,
                                          launcher=launch,
                                          k_repeats=5,
                                          timeout=60)
    assert set(outputs) == {"y"}  # only the output pointer is gathered
    assert np.allclose(outputs["y"], 3.0 * x)
    assert native_ns >= 0


def test_run_nonzero_exit_is_scored_runtimeerror(tmp_path):
    tc = c_toolchain()
    if tc is None:
        pytest.skip("no working MPI launcher in this environment")
    _cc, launch = tc
    b, desc = _yax_binding(), _descriptor()
    data = {"x": np.arange(8.0), "y": np.zeros(8), "N": 8, "a": 2.0}
    # A nonexistent executable -> the launcher exits non-zero -> a scored RuntimeError.
    with pytest.raises(RuntimeError):
        mpi_call.run(tmp_path / "no_such_bench",
                     b,
                     desc,
                     data,
                     is_python=False,
                     launcher=launch,
                     k_repeats=1,
                     timeout=30)
