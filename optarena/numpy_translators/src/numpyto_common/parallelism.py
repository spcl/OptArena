"""Timestep-loop detection carried on the IR.

A *source-form* judgement (not a dependence analysis): a ``for t in
range(TSTEPS)`` loop steps time -- each iteration depends on the last -- so it
must stay rolled (never unrolled or vectorized). The imperative backends
(C / Fortran) emit loops regardless; JAX consults this so a timestep loop lowers
to ``lax.fori_loop`` / ``while_loop`` and never unrolls.
"""
import ast
from typing import Tuple

#: Symbol-name fragments that mark a time-stepping loop bound (OptArena /
#: polybench convention). Matched case-insensitively as a substring so
#: ``TSTEPS`` / ``t_steps`` / ``NITER`` all count.
TIMESTEP_SYMBOLS: Tuple[str, ...] = (
    "TSTEPS", "TSTEP", "TIMESTEPS", "TMAX", "NITER", "NSTEPS", "NTIMESTEPS", "NTIME",
)


def _range_bound_names(node: ast.For):
    """Names referenced in the ``range(...)`` bounds of a ``for`` loop, or
    ``None`` if the loop is not a ``for x in range(...)``."""
    it = node.iter
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
            and it.func.id == "range"):
        return None
    names = set()
    for arg in it.args:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
    return names


def is_timestep_loop(node: ast.AST, timestep_symbols: Tuple[str, ...] = TIMESTEP_SYMBOLS) -> bool:
    """True when ``node`` is a ``for`` loop whose range bound references a
    time-stepping symbol (so it must stay rolled, never unrolled)."""
    if not isinstance(node, ast.For):
        return False
    names = _range_bound_names(node)
    if not names:
        return False
    syms = tuple(s.lower() for s in timestep_symbols)
    return any(s in nm.lower() for nm in names for s in syms)
