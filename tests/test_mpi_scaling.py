# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Strong + weak scaling correctness for the distributed stencils jacobi_2d / heat_3d.

test_mpi_halo checks the halo at one 1-vs-4 point; this sweeps the rank count and asserts the
gathered field is bit-exact (array_equal) with the sequential numpy oracle at every rank the
scorer runs, so a decomposition bug that only shows at a particular R is caught.

Sizes come from mpi_sizing + each manifest's mpi.decomposition block (read here, so the test can't
drift from production):
* strong -- size fixed, ranks {1,2,4}; every R must reconstruct the same oracle field.
* weak -- size grows via mpi_sizing.weak (axis x R**(1/work_exponent)); weak() only accepts
  perfect work_exponent-th-power ranks, so the sweep visits jacobi {1,4}. heat's next is R=8,
  outside {1,2,4}, so heat weak E2E is skipped.

C-only (test_mpi_halo already checks C == mpi4py); gated on an MPI toolchain, skips cleanly. The
pure sizing checks run every CI run.
"""
import numpy as np
import pytest

from optarena.agent_bench import mpi_sizing
from optarena.spec import BenchSpec
from tests.mpi_launch_helpers import c_toolchain, cc_override_for  # import sets HWLOC anti-hang env
from tests.mpi_stencil_helpers import run_stencil, seq_heat, seq_jacobi, stencil_init

# kernel -> ndim, sequential oracle, base problem. Small TSTEPS (correctness, not timing); base N
# splits cleanly over {1,2,4} and stays small once weak-grown.
STENCILS = {
    "jacobi_2d": {
        "ndim": 2,
        "seq": seq_jacobi,
        "N": 12,
        "TSTEPS": 6
    },
    "heat_3d": {
        "ndim": 3,
        "seq": seq_heat,
        "N": 8,
        "TSTEPS": 5
    },
}
STRONG_RANKS = (1, 2, 4)  # the chosen sweep envelope (no oversubscription beyond 4 ranks)
RANK_CAP = max(STRONG_RANKS)  # weak sweep shares the strong envelope; stated once


def decomposition(kernel):
    """The manifest's (axis_symbols, work_exponent) -- the values mpi_sizing uses, so weak sizing
    here matches production."""
    blk = BenchSpec.load(kernel).mpi["decomposition"]
    return list(blk["axis"]), int(blk["work_exponent"])


def weak_ranks(work_exponent, cap=RANK_CAP):
    """Ranks in 1..cap that mpi_sizing.weak accepts (perfect work_exponent-th powers). Probes weak()
    rather than re-deriving the rule, so the two can't drift."""
    ok = []
    for R in range(1, cap + 1):
        try:
            mpi_sizing.weak({"_probe": 1}, ["_probe"], R, work_exponent=work_exponent)
            ok.append(R)
        except ValueError:
            pass
    return ok


def sized(kernel, spec, mode, ranks):
    """The (N, TSTEPS) the scorer would run this kernel at for ``mode`` on ``ranks``."""
    axis, k = decomposition(kernel)
    base = {"N": spec["N"], "TSTEPS": spec["TSTEPS"]}
    return mpi_sizing.sized_params(base, mode, axis, ranks, work_exponent=k)


def c_or_skip():
    tc = c_toolchain()
    if tc is None:
        pytest.skip("no working MPI C compiler + launcher in this environment")
    return tc  # (cc, launcher)


# --------------------------------------------------------------------------------------- #
# PURE: the sizing the gated sweep relies on (no cluster; gates every CI run).
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", list(STENCILS))
def test_strong_sizing_is_rank_independent(kernel):
    """Strong scaling keeps the problem fixed: the sized params are identical for every rank
    count, so the sweep grades all ranks against one oracle."""
    spec = STENCILS[kernel]
    sizes = {tuple(sorted(sized(kernel, spec, "strong", R).items())) for R in STRONG_RANKS}
    assert len(sizes) == 1


@pytest.mark.parametrize("kernel", list(STENCILS))
def test_weak_sizing_grows_axis_by_kth_root(kernel):
    """Weak scaling grows the manifest's decomposition axis by the ``work_exponent``-th root of
    the rank count and leaves every other symbol (TSTEPS) untouched -- the contract the gated weak
    sweep depends on for a fixed per-rank work."""
    spec = STENCILS[kernel]
    axis, k = decomposition(kernel)
    assert axis == ["N"]
    for R in weak_ranks(k):
        p = sized(kernel, spec, "weak", R)
        assert p["N"] == spec["N"] * round(R**(1.0 / k))
        assert p["TSTEPS"] == spec["TSTEPS"]


# --------------------------------------------------------------------------------------- #
# GATED end-to-end: build -> scatter -> compute -> gather across a rank sweep.
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", list(STENCILS))
def test_strong_scaling_bit_exact_across_ranks(kernel):
    """Fixed size, ranks {1,2,4}. Every R must reconstruct the oracle field bit-for-bit, catching a
    rank-count-specific decomposition bug (a wrong owned band at some R), not just halo's 1-vs-4."""
    cc, launch = c_or_skip()
    spec = STENCILS[kernel]
    p = sized(kernel, spec, "strong", ranks=1)  # strong: size is R-independent
    N, TSTEPS = p["N"], p["TSTEPS"]
    ref_A, ref_B = spec["seq"](TSTEPS, *stencil_init(N, spec["ndim"]))
    for R in STRONG_RANKS:
        out = run_stencil(kernel,
                          spec["ndim"],
                          language="c",
                          launcher=launch,
                          cc_override=cc_override_for(cc),
                          N=N,
                          TSTEPS=TSTEPS,
                          R=R)
        assert np.array_equal(out["A"], ref_A), f"{kernel} strong R={R}: A != sequential oracle"
        assert np.array_equal(out["B"], ref_B), f"{kernel} strong R={R}: B != sequential oracle"


@pytest.mark.parametrize("kernel", list(STENCILS))
def test_weak_scaling_bit_exact_across_ranks(kernel):
    """Size grows with R (weak sizing); each R's result must equal the oracle at its weak-scaled
    size. Only perfect work_exponent-th-power ranks > 1 that weak() accepts are visited; heat (next
    is R=8) has none in {1,2,4}, so it skips."""
    cc, launch = c_or_skip()
    spec = STENCILS[kernel]
    _, k = decomposition(kernel)
    ranks = [R for R in weak_ranks(k) if R > 1]
    if not ranks:
        pytest.skip(f"{kernel}: no perfect {k}-th-power rank count in 2..{RANK_CAP} "
                    f"(weak scaling needs one; smallest is {2 ** k}) -- outside the sweep envelope")
    for R in ranks:
        p = sized(kernel, spec, "weak", ranks=R)
        N, TSTEPS = p["N"], p["TSTEPS"]
        ref_A, ref_B = spec["seq"](TSTEPS, *stencil_init(N, spec["ndim"]))
        out = run_stencil(kernel,
                          spec["ndim"],
                          language="c",
                          launcher=launch,
                          cc_override=cc_override_for(cc),
                          N=N,
                          TSTEPS=TSTEPS,
                          R=R)
        assert np.array_equal(out["A"], ref_A), f"{kernel} weak R={R} (N={N}): A != sequential oracle"
        assert np.array_equal(out["B"], ref_B), f"{kernel} weak R={R} (N={N}): B != sequential oracle"
