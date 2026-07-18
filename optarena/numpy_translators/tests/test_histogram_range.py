"""np.histogram(a, bins, range=(lo, hi)) must DROP samples outside [lo, hi].

Both histogram lowerings (the AST expand_histogram in lib_nodes and the string-template
_HistogramHoister in numpy_desugar) clamped an out-of-range element into bin 0 / bin-1
instead, inflating the edge bins. numpy only keeps [lo, hi] (the last bin closed).

The AST clamp's bounds are int()-wrapped so every min/max operand is int64 -- Fortran's
min(default-int, INT(.., c_int64_t)) is a mixed-kind GNU extension that -std=f2018 rejects.
"""
import numpy as np
from _op_oracle import run_op

_BACKENDS = ("c", "cpp", "fortran", "numba", "pythran")


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def test_histogram_explicit_range_drops_out_of_range():
    # a spans [-3, 5]; with range=(-2, 2) numpy keeps only -1, 0, 1 -> counts [0,1,1,1].
    # The old clamp folded -3 into bin 0 and 3, 5 into bin 3 -> [1,1,1,3].
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[:] = np.histogram(a, 4, range=(-2.0, 2.0))[0]\n")
    a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    assert np.array_equal(np.histogram(a, 4, range=(-2.0, 2.0))[0], [0, 1, 1, 1])  # numpy anchor
    res = run_op(src, "f", {"a": a}, {"out": (4, )}, {"N": 6}, shapes={"a": "(N,)", "out": "(4,)"}, backends=_BACKENDS)
    _assert_ok(res)


def test_histogram_auto_range_unchanged():
    # No explicit range: lo/hi are a.min()/a.max(), so every element is in range and the
    # guard is a no-op -- this must still match numpy (regression guard for the fix).
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[:] = np.histogram(a, 5)[0]\n")
    a = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 2.0, 2.0, 4.0])
    res = run_op(src, "f", {"a": a}, {"out": (5, )}, {"N": 8}, shapes={"a": "(N,)", "out": "(5,)"}, backends=_BACKENDS)
    _assert_ok(res)
