"""``np.einsum`` with an ``...`` ellipsis: the ellipsis is expanded to explicit
index letters from each operand's rank (in ``expand_einsum``, which has the
shape table), then lowered by the existing contraction machinery.

The plain-subscript parser stays ellipsis-free -- its
``test_parse_einsum_ellipsis_unsupported`` guard is intact; only the expander
does the rank-aware expansion.
"""
import numpy as np
import pytest
from _op_oracle import run_op

from numpyto_common.lib_nodes import _expand_einsum_ellipsis

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res, label):
    fails = {b: s for b, s in res.items() if not (s == "ok" or s.startswith("skip"))}
    assert not fails, f"{label}: {fails}"


def test_expand_einsum_ellipsis_to_explicit_letters():
    # One broadcast axis -> a fresh shared index letter; the output ellipsis
    # expands to the same letters (ellipsis-first, as numpy orders it).
    assert _expand_einsum_ellipsis("...ij->...ji", [3]) == "Aij->Aji"
    assert _expand_einsum_ellipsis("...ij,...jk->...ik", [3, 3]) == "Aij,Ajk->Aik"
    # Two broadcast axes (rank-4 operand, two explicit indices).
    assert _expand_einsum_ellipsis("...ij->...ji", [4]) == "ABij->ABji"


def test_expand_einsum_ellipsis_rejects_bad_forms():
    with pytest.raises(NotImplementedError):  # implicit output not supported with ellipsis
        _expand_einsum_ellipsis("...ij", [3])
    with pytest.raises(NotImplementedError):  # differing ellipsis ranks (broadcast) unsupported
        _expand_einsum_ellipsis("...ij,...jk->...ik", [4, 3])


def test_einsum_ellipsis_batched_transpose_e2e():
    rng = np.random.default_rng(0)
    src = "import numpy as np\ndef f(a, out):\n    out[:] = np.einsum('...ij->...ji', a)\n"
    a = rng.random((2, 3, 4))
    _assert_ok(
        run_op(src,
               "f", {"a": a}, {"out": (2, 4, 3)}, {
                   "B": 2,
                   "M": 3,
                   "N": 4
               },
               shapes={
                   "a": "(B, M, N)",
                   "out": "(B, N, M)"
               },
               backends=_NATIVE), "einsum-ellipsis-transpose")


def test_einsum_ellipsis_batched_matmul_e2e():
    # Ellipsis WITH a contraction: '...ij,...jk->...ik' expands to 'Aij,Ajk->Aik'
    # (a batched GEMM). Native c/cpp only: the batched-einsum Fortran emit
    # non-deterministically mistypes a size symbol as REAL (a pre-existing
    # hash/order-dependent emit bug, tracked in BACKLOG -- reproduces on the
    # EXPLICIT 'Bij,Bjk->Bik' too, so it is not the ellipsis expansion).
    rng = np.random.default_rng(0)
    src = "import numpy as np\ndef f(a, b, out):\n    out[:] = np.einsum('...ij,...jk->...ik', a, b)\n"
    a, b = rng.random((2, 3, 4)), rng.random((2, 4, 5))
    _assert_ok(
        run_op(src,
               "f", {
                   "a": a,
                   "b": b
               }, {"out": (2, 3, 5)}, {
                   "B": 2,
                   "M": 3,
                   "K": 4,
                   "N": 5
               },
               shapes={
                   "a": "(B, M, K)",
                   "b": "(B, K, N)",
                   "out": "(B, M, N)"
               },
               backends=("c", "cpp")), "einsum-ellipsis-matmul")
