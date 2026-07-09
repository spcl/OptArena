# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end translation gate for a random sample of KernelBench ports.

The ML/kernelbench track is large and its kernels are not swept by the
foundation/hpc e2e gate (tests/test_e2e_numerical.py). This module pins a
FIXED random sample -- 10 level1 + 10 level2 kernels drawn with a fixed seed --
and asserts each stays bit-exact end to end: emitted + run on every backend that
can lower it, compared against the numpy reference.

A backend that genuinely cannot lower a kernel reports ``skip:*`` or
``FAIL:emit`` / ``FAIL:compile`` -- an accepted translator gap, recorded here
per (kernel, backend) in ``_KNOWN_GAPS`` so it is visible, never silently
dropped. What is NOT tolerated is a NUMERICAL mismatch (``FAIL:<out>:d=...``):
the translation is meant to be bit-exact with numpy, so any nonzero discrepancy
is a real bug and fails the gate.

The sample is regenerated with scripts/select_kernelbench_sample.py; edit the
lists below only via that script so the selection stays reproducible.
"""
import pytest

from tests.numerical_oracle import run_kernel

# Fixed random sample (seed=0). Regenerate via scripts/select_kernelbench_sample.py.
SELECTED = {
    # level1: native-green (c/cpp/fortran bit-exact) matmul/norm/activation kernels.
    "level1": [
        "four_d_tensor_matrix_multiplication",
        "three_d_tensor_matrix_multiplication",
        "batched_matrix_multiplication",
        "matmul_with_transposed_a",
        "matmul_with_small_k_dimension",
        "frobenius_norm",
        "l1_norm",
        "l2_norm",
        "gelu",
        "relu",
    ],
    # level2: composite kernels; numba/jax bit-exact (native emit is a tracked gap
    # for the conv/reshape families -- see _KNOWN_GAPS).
    "level2": [
        "conv3d_group_norm_mean",
        "conv3d_mish_tanh",
        "conv3d_min_softmax",
        "conv3d_scaling_tanh_multiply_sigmoid",
        "conv3d_max_logsumexp_relu",
        "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add",
        "conv2d_relu_bias_add",
        "conv2d_hardswish_relu",
        "gemm_group_norm_swish_multiply_swish",
        "matmul_swish_sum_group_norm",
    ],
}

# (kernel, backend) pairs where the backend cannot lower the op today -- an
# accepted, tracked translator gap (never a numerical mismatch).
_KNOWN_GAPS: set = set()

_BACKENDS = ("c", "cpp", "fortran", "numba", "jax")
_CACHE: dict = {}


def _result(name: str) -> dict:
    if name not in _CACHE:
        _CACHE[name] = run_kernel(name, "S", "fp64", seed=0, only_backends=frozenset(_BACKENDS))
    return _CACHE[name]


def _params():
    for level, names in SELECTED.items():
        for name in names:
            yield pytest.param(name, id=f"{level}-{name}")


@pytest.mark.parametrize("name", list(_params()))
def test_kernelbench_translation_bit_exact(name):
    status = _result(name)
    # A numerical mismatch on ANY backend is a real bug.
    mism = {b: s for b, s in status.items() if s.startswith("FAIL") and ":d=" in s}
    assert not mism, f"{name}: numerical mismatch (translation not bit-exact): {mism}"
    # At least one backend must have actually run bit-exact (else the kernel is
    # untested -- a silent gap the whole point of this gate is to prevent).
    oks = [b for b, s in status.items() if s == "ok"]
    assert oks, f"{name}: no backend validated it (all skip/emit-gap): {status}"
