# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel timeout: the resolver precedence + the runner's finalize/timeout wiring.

``resolve_kernel_timeout`` picks the per-kernel agent-run budget by precedence
(global override > kernel-yaml ``timeout_s`` > per-level default > flat fallback),
and ``solve_task`` runs the kernel in a forked child bounded by that budget -- an
overrun ends the run with a scored ``timeout`` row (never a hang).
"""
import time
import types

import pytest

from optarena import config
from optarena.agent_bench.agent import StubAgent
from optarena.agent_bench.runner import solve_task
from optarena.agent_bench.scoring import resolve_kernel_timeout
from optarena.agent_bench.task import Task
from optarena.spec import BenchSpec


def _spec(*, level=None, **extra):
    """A minimal stand-in for a BenchSpec: ``resolved_level`` + whatever manifest
    fields (e.g. ``timeout_s``) the resolver reads via ``vars(spec).get``. A real
    frozen BenchSpec exposes ``timeout_s`` the same way once the schema carries it."""
    return types.SimpleNamespace(resolved_level=level, **extra)


@pytest.fixture
def pinned_timeouts():
    """Pin the timeout config to known values (independent of config.yaml edits),
    cleared afterwards."""
    config.set_override("timeouts.kernel_s", 300)
    config.set_override("timeouts.kernel_s_by_level", {1: 11, 2: 22, 3: 33})
    config.set_override("timeouts.kernel_s_override", None)
    try:
        yield
    finally:
        for k in ("timeouts.kernel_s", "timeouts.kernel_s_by_level", "timeouts.kernel_s_override"):
            config.clear_override(k)


def test_override_wins_over_everything(pinned_timeouts):
    config.set_override("timeouts.kernel_s_override", 42)
    # even with a kernel-yaml timeout_s AND a matching per-level default, the global override wins
    assert resolve_kernel_timeout(_spec(level=1, timeout_s=123)) == 42.0


def test_kernel_yaml_wins_over_level_and_fallback(pinned_timeouts):
    assert resolve_kernel_timeout(_spec(level=1, timeout_s=123)) == 123.0
    # no level either -> still the kernel-yaml value, not the flat fallback
    assert resolve_kernel_timeout(_spec(level=None, timeout_s=123)) == 123.0


def test_per_level_default_when_no_override_or_yaml(pinned_timeouts):
    assert resolve_kernel_timeout(_spec(level=1)) == 11.0
    assert resolve_kernel_timeout(_spec(level=2)) == 22.0
    assert resolve_kernel_timeout(_spec(level=3)) == 33.0


def test_fallback_for_none_level_or_unmapped_level(pinned_timeouts):
    # None level falls through to the flat fallback ...
    assert resolve_kernel_timeout(_spec(level=None)) == 300.0
    # ... as does a level with no per-level entry.
    config.set_override("timeouts.kernel_s_by_level", {1: 11})
    assert resolve_kernel_timeout(_spec(level=2)) == 300.0


def test_string_keyed_level_map_is_tolerated(pinned_timeouts):
    """An env/JSON-sourced by-level map may key levels as strings; the int level still matches."""
    config.set_override("timeouts.kernel_s_by_level", {"2": 77})
    assert resolve_kernel_timeout(_spec(level=2)) == 77.0


def test_real_benchspec_has_no_timeout_s_and_uses_its_level(pinned_timeouts):
    """A real (frozen) BenchSpec carries no ``timeout_s`` field today: the resolver
    reads it as absent (not an error) and falls to the per-level default. gemm is L1."""
    spec = BenchSpec.load("gemm")
    assert spec.resolved_level == 1
    assert resolve_kernel_timeout(spec) == 11.0  # by_level[1] from the fixture


# -- the runner wiring: an overrun ends the run with a scored timeout row --------


class _HangAgent(StubAgent):
    """Never returns a submission -- exercises the per-kernel timeout path."""
    name = "hang"

    def solve(self, task, prompt="", budget=None):
        while True:
            time.sleep(0.05)


def test_solve_task_times_out_to_a_scored_row():
    """A hanging agent run is bounded by the per-kernel budget and recorded as a
    scored ``timeout`` row -- the runner never hangs, and best-so-far (none here)
    stands as not-solved."""
    row, sub = solve_task(_HangAgent(), Task("gemm", "restricted", "c"), timeout=1.0)
    assert row.status == "timeout" and row.correct is False and sub is None
    assert "time" in row.detail.lower()
