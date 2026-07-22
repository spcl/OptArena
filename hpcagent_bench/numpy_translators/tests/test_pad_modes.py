"""``np.pad(a, w, mode=...)`` reflect / wrap / symmetric -> a per-axis boundary
index remap, lowered by ``expand_pad`` (the native c / c++ / fortran path).

Each output cell reads the source cell whose index folds ``out - before`` back
into ``[0, d)``: ``wrap`` = periodic (mod d), ``symmetric`` = mirror INCLUDING
the edge (period 2d), ``reflect`` = mirror EXCLUDING the edge (period 2(d-1)).

Validated bit-exact vs numpy, including a pad width larger than the axis
(multi-period wraparound) and BOTH a symbolic-int64 extent (period bound to an
int local) and a literal extent (period folded to a literal) -- the modulus must
match the int64 index kind under Fortran's kind-strict MODULO.
"""
import numpy as np
import pytest
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res, label):
    fails = {b: s for b, s in res.items() if not (s == "ok" or s.startswith("skip"))}
    assert not fails, f"{label}: {fails}"


@pytest.mark.parametrize("mode", ["reflect", "wrap", "symmetric"])
@pytest.mark.parametrize("n,w", [(6, 2), (4, 5)])  # w > n exercises the multi-period remap
@pytest.mark.parametrize("symbolic", [True, False])
def test_pad_boundary_mode(mode, n, w, symbolic):
    src = (f"import numpy as np\n"
           f"def pad_op(a, out):\n"
           f"    out[:] = np.pad(a, {w}, mode='{mode}')\n")
    a = np.random.default_rng(0).random((n, ))
    out_shape = (n + 2 * w, )
    label = f"pad-{mode}-n{n}-w{w}-{'sym' if symbolic else 'lit'}"
    if symbolic:
        res = run_op(src,
                     "pad_op", {"a": a}, {"out": out_shape}, {"N": n},
                     shapes={
                         "a": "(N,)",
                         "out": f"(N + {2 * w},)"
                     },
                     backends=_NATIVE)
    else:
        res = run_op(src, "pad_op", {"a": a}, {"out": out_shape}, {}, backends=_NATIVE)
    _assert_ok(res, label)


def test_pad_reflect_size1_axis_repeats():
    # reflect on a size-1 axis has period 0 in numpy -> it just repeats the one
    # element; the lowering guards this (no modulo-by-zero) and returns index 0.
    src = "import numpy as np\ndef pad_op(a, out):\n    out[:] = np.pad(a, 2, mode='reflect')\n"
    a = np.array([7.0])
    _assert_ok(run_op(src, "pad_op", {"a": a}, {"out": (5, )}, {}, backends=_NATIVE), "pad-reflect-size1")
