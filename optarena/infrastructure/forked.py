# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a callable in a forked child and SURFACE its failure instead of eating it.

Native (no-container) collection runs each kernel in its own child process so a
segfault or a framework exception in one kernel cannot take down the whole sweep --
and, unlike a swallow-everything harness, the cause is reported: a fatal signal
(``SIGSEGV`` / ``SIGABRT`` / ... from a crashing kernel) is decoded to its name, any
Python traceback is printed to stdout by the child before it exits, and a child that
runs past ``timeout`` seconds is terminated and reported as ``TIMEOUT``.

This is the shared isolation primitive for the native framework-baseline collection
and the native agent run (and the per-kernel wall-clock budget).
"""
import multiprocessing
import queue
import signal
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

#: Grace period (seconds) to drain the result queue after the child has exited
#: cleanly -- the process is done but the queue feeder thread may still be flushing.
_DRAIN_S = 5.0


@dataclass
class RunResult:
    """Outcome of a forked run. ``ok`` is the only success signal; on failure exactly
    one of ``signal`` (fatal signal name, e.g. ``SIGSEGV`` / ``TIMEOUT``) or ``error``
    (a traceback / message) explains why. ``result`` carries the callable's return
    value on success (must be picklable; ``None`` if it was not)."""
    ok: bool
    exit_code: Optional[int] = None
    signal: Optional[str] = None
    error: Optional[str] = None
    result: Any = None


def _child(fn, args, kwargs, q):
    try:
        out = fn(*args, **kwargs)
        try:
            q.put(("ok", out))
        except Exception:  # unpicklable return value -> success without a payload
            q.put(("ok", None))
    except BaseException:  # noqa: BLE001 -- surface EVERY failure, never swallow it
        tb = traceback.format_exc()
        sys.stdout.write(tb)
        sys.stdout.flush()
        q.put(("error", tb))


def run_forked(fn: Callable, *args, label: str = "", timeout: Optional[float] = None, **kwargs) -> RunResult:
    """Run ``fn(*args, **kwargs)`` in a forked child.

    Returns a failed :class:`RunResult` -- and logs the cause to stdout -- on a fatal
    signal (segfault), an exception (traceback), or a ``timeout`` overrun; otherwise
    ``ok=True`` with the (picklable) return value. ``timeout`` is a per-call wall-clock
    budget in seconds (``None`` waits forever); ``label`` tags the stdout log lines."""
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(fn, args, kwargs, q))
    tag = f"[{label}] " if label else ""
    p.start()
    p.join(timeout)
    if p.is_alive():  # ran past the budget -- kill it and report the timeout
        p.terminate()
        p.join()
        msg = f"{tag}timed out after {timeout}s"
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
        return RunResult(ok=False, signal="TIMEOUT", error=msg)
    ec = p.exitcode
    if ec is not None and ec < 0:  # killed by a fatal signal (segfault, abort, ...)
        try:
            sig = signal.Signals(-ec).name
        except ValueError:
            sig = f"signal {-ec}"
        msg = f"{tag}child killed by {sig}"
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
        return RunResult(ok=False, exit_code=ec, signal=sig, error=msg)
    try:
        status, payload = q.get(timeout=_DRAIN_S)
    except queue.Empty:
        return RunResult(ok=False, exit_code=ec, error=f"{tag}child exited {ec} with no result")
    if status == "ok":
        return RunResult(ok=True, exit_code=ec, result=payload)
    return RunResult(ok=False, exit_code=ec, error=payload)
