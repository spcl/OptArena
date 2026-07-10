"""Timestep-loop detection (the IR annotation JAX consumes to keep a time-stepping
loop rolled). Pure-logic unit test; imports resolve via PYTHONPATH."""
import ast

from numpyto_common.parallelism import is_timestep_loop


def _stmt(src):
    return ast.parse(src).body[0]


def test_timestep_for_is_flagged():
    assert is_timestep_loop(_stmt("for t in range(TSTEPS):\n    step(t)\n"))


def test_plain_range_is_not_timestep():
    assert not is_timestep_loop(_stmt("for i in range(N):\n    a[i] = 0\n"))


def test_niter_counts_as_timestep():
    assert is_timestep_loop(_stmt("for k in range(NITER):\n    pass\n"))


def test_non_for_node_is_not_timestep():
    assert not is_timestep_loop(_stmt("a[1:-1] = b[:-2] + b[2:]\n"))
