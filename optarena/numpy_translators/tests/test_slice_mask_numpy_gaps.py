"""NumPy-op gaps surfaced auditing the NumpyToC surface (PR#5 follow-up).

Four families, each validated numerically vs numpy across C / C++ / Fortran:

* ``a[::-1]`` reverse (and general strided) slices -- the step was dropped, so a
  reverse copy emitted a plain forward copy;
* boolean-mask REDUCTIONS ``np.sum/mean/max/min(a[m])`` inline (the peephole only
  matched the two-statement ``t = a[m]; np.sum(t)`` form, and the masked select
  otherwise materialised as a wrong integer gather ``a[m[i]]``);
* ``np.any`` / ``np.all`` / ``np.count_nonzero`` -- the bool-as-int accumulator
  (``acc + (x != 0)``) is invalid Fortran arithmetic; rebuilt as an if-guarded
  integer accumulator, and their non-Name arg now hoists like ``sum``;
* ``~mask`` (bitwise-not on a boolean array) -- numpy logical negation, but both
  backends emitted integer bitwise NOT (``~1`` -> the truthy ``-2`` in C, an
  ``i`` argument error in Fortran).
"""
import numpy as np
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def _run(body, ins, outs, syms, shapes):
    src = "import numpy as np\ndef f(" + ", ".join(list(ins) + list(outs)) + "):\n" + body + "\n"
    return _ok(run_op(src, "f", ins, outs, syms, shapes=shapes, backends=_NATIVE))


_A = np.linspace(0.5, 3.0, 6)


# --- reverse / strided slices --------------------------------------------- #


def test_reverse_whole_array_assign():
    ok, res = _run(" out[:] = a[::-1]", {"a": _A}, {"out": (6, )}, {"N": 6}, {"a": "(N,)", "out": "(N,)"})
    assert ok, res


def test_reverse_to_local_then_copy():
    ok, res = _run(" b = a[::-1]\n for i in range(6):\n  out[i] = b[i]",
                   {"a": _A}, {"out": (6, )}, {"N": 6}, {"a": "(N,)", "out": "(N,)"})
    assert ok, res


def test_strided_reverse_step2():
    ok, res = _run(" b = a[::-2]\n for i in range(3):\n  out[i] = b[i]",
                   {"a": _A}, {"out": (3, )}, {"N": 6}, {"a": "(N,)", "out": "(3,)"})
    assert ok, res


def test_forward_strided_slice_regression():
    ok, res = _run(" b = a[1:6:2]\n for i in range(3):\n  out[i] = b[i]",
                   {"a": _A}, {"out": (3, )}, {"N": 6}, {"a": "(N,)", "out": "(3,)"})
    assert ok, res


def test_whole_array_copy_regression():
    ok, res = _run(" out[:] = a[:]", {"a": _A}, {"out": (6, )}, {"N": 6}, {"a": "(N,)", "out": "(N,)"})
    assert ok, res


# --- boolean-mask reductions ---------------------------------------------- #


def test_masked_sum_inline():
    ok, res = _run(" m = a > 1.5\n out[0] = np.sum(a[m])", {"a": _A}, {"out": (1, )}, {"N": 6},
                   {"a": "(N,)", "out": "(1,)"})
    assert ok, res


def test_masked_mean_inline():
    ok, res = _run(" m = a > 1.5\n out[0] = np.mean(a[m])", {"a": _A}, {"out": (1, )}, {"N": 6},
                   {"a": "(N,)", "out": "(1,)"})
    assert ok, res


def test_masked_max_min_method_form():
    for op in ("max", "min"):
        ok, res = _run(f" m = a > 1.5\n out[0] = a[m].{op}()", {"a": _A}, {"out": (1, )}, {"N": 6},
                       {"a": "(N,)", "out": "(1,)"})
        assert ok, (op, res)


def test_masked_max_seed_excludes_index0():
    """Index 0 is masked OUT and holds the global max; the accumulator must seed
    from the first masked HIT, not ``arr[0]`` (correct masked-max = 3.0)."""
    a = np.array([10.0, 1.0, 2.0, 3.0])
    ok, res = _run(" m = a < 5.0\n out[0] = np.max(a[m])", {"a": a}, {"out": (1, )}, {"N": 4},
                   {"a": "(N,)", "out": "(1,)"})
    assert ok, res


def test_integer_gather_reduction_not_masked():
    """``np.sum(a[idx])`` with an INTEGER index array is a gather-sum, NOT a
    masked reduction -- it must stay a gather (the mask peephole is gated on a
    known-boolean index)."""
    a = np.linspace(0.5, 3.0, 6)
    ok, res = _run(" out[0] = np.sum(a[idx])", {"a": a, "idx": np.array([0, 2, 4])}, {"out": (1, )},
                   {"N": 6}, {"a": "(N,)", "idx": "(3,)", "out": "(1,)"})
    assert ok, res


# --- any / all / count_nonzero -------------------------------------------- #


def test_any_all_named_mask():
    for op, expect_body in (("any", " m = a > 1.5\n"), ("all", " m = a > 0.0\n")):
        ok, res = _run(expect_body + f" out[0] = np.{op}(m)", {"a": _A}, {"out": (1, )}, {"N": 6},
                       {"a": "(N,)", "out": "(1,)"})
        assert ok, (op, res)


def test_count_nonzero_mask_and_float():
    ok, res = _run(" m = a > 1.5\n out[0] = np.count_nonzero(m)", {"a": _A}, {"out": (1, )}, {"N": 6},
                   {"a": "(N,)", "out": "(1,)"})
    assert ok, res
    x = np.array([0.0, 1.5, 0.0, 2.0, 3.0, 0.0])
    ok, res = _run(" out[0] = np.count_nonzero(a)", {"a": x}, {"out": (1, )}, {"N": 6},
                   {"a": "(N,)", "out": "(1,)"})
    assert ok, res


def test_any_all_count_inline_compare():
    for op in ("any", "all", "count_nonzero"):
        ok, res = _run(f" out[0] = np.{op}(a > 1.5)", {"a": _A}, {"out": (1, )}, {"N": 6},
                       {"a": "(N,)", "out": "(1,)"})
        assert ok, (op, res)


def test_count_nonzero_axis():
    a = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    ok, res = _run(" out[:] = np.count_nonzero(a, axis=1)", {"a": a}, {"out": (2, )}, {"M": 2, "N": 3},
                   {"a": "(M, N)", "out": "(M,)"})
    assert ok, res


def test_any_all_axis():
    a = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    ok, res = _run(" m = a > 1.0\n out[:] = np.any(m, axis=0)", {"a": a}, {"out": (3, )}, {"M": 2, "N": 3},
                   {"a": "(M, N)", "out": "(N,)"})
    assert ok, res
    ok, res = _run(" m = a >= 0.0\n out[:] = np.all(m, axis=1)", {"a": a}, {"out": (2, )}, {"M": 2, "N": 3},
                   {"a": "(M, N)", "out": "(M,)"})
    assert ok, res


# --- ~ (bitwise-not on a boolean mask = logical negation) ----------------- #

_S = np.linspace(-1.0, 3.0, 6)


def test_not_mask_in_where():
    ok, res = _run(" m = a > 1.0\n out[:] = np.where(~m, a, 0.0)", {"a": _S}, {"out": (6, )}, {"N": 6},
                   {"a": "(N,)", "out": "(N,)"})
    assert ok, res


def test_not_mask_inline_compare():
    ok, res = _run(" out[:] = np.where(~(a > 1.0), a, 0.0)", {"a": _S}, {"out": (6, )}, {"N": 6},
                   {"a": "(N,)", "out": "(N,)"})
    assert ok, res


def test_not_mask_named_then_reduce():
    ok, res = _run(" m = a > 1.0\n nm = ~m\n out[0] = np.sum(a[nm])", {"a": _S}, {"out": (1, )}, {"N": 6},
                   {"a": "(N,)", "out": "(1,)"})
    assert ok, res


def test_not_mask_in_combine():
    for combine in ("m & ~m3", "m | ~m3", "~m & ~m3"):
        ok, res = _run(f" m = a > 1.0\n m3 = a > 2.0\n out[:] = np.where({combine}, a, 0.0)",
                       {"a": _S}, {"out": (6, )}, {"N": 6}, {"a": "(N,)", "out": "(N,)"})
        assert ok, (combine, res)
