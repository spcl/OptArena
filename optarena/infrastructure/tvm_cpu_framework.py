# Copyright 2025 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""TVM CPU-side helpers for the unified TVM framework.

Holds the ``tvm_dtype`` string kernels read for their ``te.placeholder`` shapes,
the datatype→TVM-dtype mapping, and the MetaSchedule ``tune_tir`` trial budget
(``tvm.compile`` codegen, Apache TVM 0.20+; the old ``tvm.auto_scheduler`` path is
unsupported). ``TVMCPUFramework`` is a back-compat alias of the one arch-branched
:class:`~optarena.infrastructure.tvm_framework.TVMFramework` (dace_cpu/dace_gpu ->
one class).
"""

# Datatype string picked by the harness's set_datatype(). Kernel files
# read this when constructing their te.placeholder shapes.
tvm_dtype: str = "float64"


def tvm_dtype_str(datatype) -> str:
    """The TVM dtype string for a datatype request (numpy or enum spelling).

    fp64/fp32/fp16/bf16 map to their TVM names; anything else (fp8 -- TVM's
    support is partial and the siblings pin float32 there) falls back to float64.
    """
    from optarena.precision import Precision, precision_from_datatype
    return {
        Precision.FP64: "float64",
        Precision.FP32: "float32",
        Precision.FP16: "float16",
        Precision.BF16: "bfloat16",
    }.get(precision_from_datatype(datatype), "float64")


# Per-process trial cap. Smaller = faster sanity smoke; full = paper.
METASCHEDULE_TRIALS_DEFAULT = 64
METASCHEDULE_TRIALS_FULL = 1024


def metaschedule_trials() -> int:
    """How many tuning trials to give ``tune_tir`` per task.

    Delegates to the unified optimize budget (:class:`optarena.optimize.\
OptimizeBudget`) so TVM and Triton share ONE knob (``$OPTARENA_OPTIMIZE_BUDGET``),
    read on every call so a test can change it mid-process."""
    from optarena.optimize import OptimizeBudget
    return OptimizeBudget.from_env().tvm_trials()


# TVM CPU and GPU are one arch-branched framework now: ``TVMCPUFramework`` is a
# back-compat alias of :class:`TVMFramework` (imported after the helpers above so a
# ``from optarena.infrastructure.tvm_cpu_framework import *`` still re-exports it, and
# consumers that reference ``TVMCPUFramework`` keep working). tvm_framework imports
# this module only lazily (inside set_datatype), so this top-level import is one-way.
from optarena.infrastructure.tvm_framework import TVMFramework as TVMCPUFramework  # noqa: E402
