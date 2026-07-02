"""Emit a DaCe ``@dc.program`` from the canonical numpy reference.

The Foundation-track numpy kernels are already dace-shaped (they were
ported FROM ``@dace.program`` bodies), and they now carry every size
symbol as an explicit scalar argument plus a per-array shape. That is
exactly the metadata a numpy->dace converter needs:

* size SYMBOLS (``LEN_1D``, ``K``, ...) become module-level
  ``dc.symbol(...)`` declarations and are DROPPED from the program
  signature (dace passes them implicitly via the array shapes);
* ARRAY arguments are typed ``<dctype>[shape]`` -- ``dc_float`` for the
  precision-driven floats, ``dc.int32`` / ``dc.int64`` for index arrays
  (the dtype the original ``dace.int*`` annotation carried, recovered
  from bench_info ``init.dtypes``);
* kernel SCALARS (``alpha``, s174's ``M``, ...) stay typed scalar args.

The body is the numpy reference verbatim. Reuses :func:`parse_kernel`
so the array/symbol/scalar classification is the single source of
truth shared with the C/Fortran emitters.
"""

import ast
import copy
import re
from typing import Dict, List

from numpyto_c.ir import KernelIR
from numpyto_common.numpy_desugar import desugar_for_python_backend

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


class _ShapeToSymbol(ast.NodeTransformer):
    """dace has no runtime ``.shape``: an array carries a SYMBOLIC shape (the
    ``a: dc_float[M + 1]`` annotation), so ``a.shape[k]`` IS that symbolic dimension.
    Replace each ``<array>.shape[<const k>]`` with the array's k-th declared shape
    token (``A_indptr.shape[0]`` -> ``M + 1``)."""

    def __init__(self, arr_shapes: Dict[str, List[str]]):
        self.arr_shapes = arr_shapes

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        v = node.value
        if (isinstance(v, ast.Attribute) and v.attr == "shape" and isinstance(v.value, ast.Name)
                and v.value.id in self.arr_shapes and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)):
            toks = self.arr_shapes[v.value.id]
            if 0 <= node.slice.value < len(toks):
                return ast.copy_location(ast.parse(toks[node.slice.value], mode="eval").body, node)
        return node


class _DropSymbolAssign(ast.NodeTransformer):
    """Drop ``<sym> = ...`` where ``<sym>`` is a declared size symbol. A dc.symbol is a
    compile-time constant the harness supplies via the array shapes, so recomputing it
    in the body (``M = A_indptr.shape[0] - 1``) is both redundant and illegal in dace."""

    def __init__(self, symbols):
        self.symbols = set(symbols)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.symbols:
            return None
        return node

#: numpy dtype tag -> dace type expression. Floats route through the
#: precision-driven ``dc_float`` / ``dc_complex_float`` globals the dace
#: framework rebinds per run; integers carry a fixed width.
_DTYPE_TO_DACE = {
    "float64": "dc_float",
    "float32": "dc_float",
    "complex128": "dc_complex_float",
    "complex64": "dc_complex_float",
    "int64": "dc.int64",
    "int32": "dc.int32",
    "int16": "dc.int16",
    "int8": "dc.int8",
    "uint64": "dc.uint64",
    "uint32": "dc.uint32",
    "uint16": "dc.uint16",
    "uint8": "dc.uint8",
    "int": "dc.int64",
    "bool": "dc.bool",
}


def _dace_dtype(tag: str) -> str:
    return _DTYPE_TO_DACE.get(tag, "dc_float")


def _array_annotation(arr) -> str:
    """``a`` of shape ``(LEN_1D,)`` float64 -> ``dc_float[LEN_1D]``."""
    shape = ", ".join(str(s) for s in arr.shape) if arr.shape else "1"
    return f"{_dace_dtype(arr.dtype)}[{shape}]"


def emit_dace(kir: KernelIR, fn_name: str | None = None) -> str:
    """Return the source of a ``<short>_dace.py`` module for ``kir``."""
    name = fn_name or kir.kernel_name
    arrays = {a.name: a for a in kir.arrays}
    scalars = {s.name: s for s in kir.scalars}
    symbol_names = [s.name for s in kir.symbols]
    # Sparse kirs carry their size symbols (nnz, M, N) ONLY in the array shapes, not
    # kir.symbols -- collect every free identifier in a shape that is not itself an
    # array or scalar, so it is declared as a dc.symbol below (otherwise the shape
    # annotation ``A_indptr: dc.int64[M + 1]`` references an undefined name).
    arr_shapes = {a.name: [str(s) for s in a.shape] for a in kir.arrays}
    _known = set(arrays) | set(scalars)
    for _toks in arr_shapes.values():
        for _tok in _toks:
            for _ident in _IDENT_RE.findall(_tok):
                if _ident not in _known and _ident not in symbol_names:
                    symbol_names.append(_ident)

    # Program signature: arrays + scalars in their original input_args
    # order; symbols are module-level, not parameters.
    params: List[str] = []
    for arg in kir.input_args:
        if arg in arrays:
            params.append(f"{arg}: {_array_annotation(arrays[arg])}")
        elif arg in scalars:
            params.append(f"{arg}: {_dace_dtype(scalars[arg].dtype)}")
        # symbols: skip (declared at module scope below)

    needs_complex = any(_dace_dtype(a.dtype) == "dc_complex_float"
                        for a in kir.arrays) or any(_dace_dtype(s.dtype) == "dc_complex_float" for s in kir.scalars)

    out: List[str] = []
    out.append('"""DaCe program auto-generated from the numpy reference '
               'by numpyto_c.dace_emit."""')
    out.append("import numpy as np")
    out.append("import dace as dc")
    imp = "dc_float, dc_complex_float" if needs_complex else "dc_float"
    out.append(f"from optarena.infrastructure.dace_framework import {imp}")
    out.append("from math import sin, cos, log, exp, pow")
    out.append("")
    if symbol_names:
        names = ", ".join(symbol_names)
        srcs = ", ".join(f"'{s}'" for s in symbol_names)
        if len(symbol_names) == 1:
            out.append(f"{names} = dc.symbol({srcs}, dtype=dc.int64)")
        else:
            out.append(f"{names} = (dc.symbol(s, dtype=dc.int64) "
                       f"for s in ({srcs}))")
        out.append("")
    out.append("")
    out.append("@dc.program")
    out.append(f"def {name}({', '.join(params)}):")

    # Body: the numpy reference desugared for the verbatim-body backends -- the
    # SAME pass numba/pythran use, so dace gains feature parity (np.fft, fancy
    # multi-index gather, np.add.at scatter, axis reductions, boolean-mask
    # assignment, ... lower to the plain loops a @dc.program traces). The
    # function is renamed to kir.kernel_name first so the desugar seeds its rank/
    # dtype tables from the kir arrays; falls back to the verbatim body if the
    # desugar cannot parse (it never should for a valid kernel).
    fn_ast = copy.deepcopy(kir.tree)
    fn_ast.name = kir.kernel_name
    try:
        desugared = desugar_for_python_backend(ast.unparse(fn_ast), kir, backend="dace")
        fn_ast = next(n for n in ast.parse(desugared).body if isinstance(n, ast.FunctionDef))
    except Exception:  # noqa: BLE001 -- keep the verbatim body if desugar fails
        fn_ast = kir.tree
    # dace has no runtime ``.shape`` and its symbols are immutable: rewrite
    # ``arr.shape[k]`` to the symbolic dimension and drop any recompute of a size
    # symbol (``M = A_indptr.shape[0] - 1``), which is redundant + illegal in dace.
    fn_ast = _ShapeToSymbol(arr_shapes).visit(fn_ast)
    fn_ast = _DropSymbolAssign(symbol_names).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    body = list(fn_ast.body)
    if (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body:
        out.append("    pass")
    else:
        for stmt in body:
            for line in ast.unparse(stmt).splitlines():
                out.append("    " + line)
    return "\n".join(out) + "\n"
