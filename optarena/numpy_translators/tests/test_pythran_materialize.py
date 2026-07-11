"""Pythran cannot reshape a non-materialized object nor index a lazy
``numpy_expr`` passed into a helper (KernelBench lenet/mlp). ``_PythranMaterialize``
forces evaluation with ``np.ascontiguousarray``; these AST tests pin the rewrite,
and the end-to-end bit-exact numba/pythran validation lives in the ml oracle.
"""
import ast

from numpyto_pythran.emit import _PythranMaterialize


def _apply(src, local_funcs):
    tree = ast.parse(src)
    tree = _PythranMaterialize(set(local_funcs)).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def test_np_reshape_becomes_ascontiguous_method_reshape():
    # the shape tuple is preserved as the reshape method arg (pythran accepts
    # ``.reshape((N, C))``; validated end-to-end by lenet's pythran gate).
    out = _apply("y = np.reshape(x, (N, C))", local_funcs=[])
    assert out == "y = np.ascontiguousarray(x).reshape((N, C))"


def test_helper_call_compound_arg_is_materialized():
    out = _apply("y = softmax(x @ w + b)", local_funcs=["softmax"])
    assert out == "y = softmax(np.ascontiguousarray(x @ w + b))"


def test_helper_call_plain_name_arg_untouched():
    # a bare Name is already concrete -> no ascontiguousarray wrap.
    out = _apply("y = softmax(t)", local_funcs=["softmax"])
    assert out == "y = softmax(t)"


def test_non_local_call_arg_untouched():
    # np.exp is not a local helper; its lazy arg is fine (elementwise).
    out = _apply("y = np.exp(x - m)", local_funcs=["softmax"])
    assert out == "y = np.exp(x - m)"


def test_method_reshape_not_double_wrapped():
    # an existing ``x.reshape(...)`` method form is left alone (only the
    # ``np.reshape`` function form is the failing shape).
    out = _apply("y = x.reshape(N, C)", local_funcs=[])
    assert out == "y = x.reshape(N, C)"
