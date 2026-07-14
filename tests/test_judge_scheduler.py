# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The judge device model: DeviceSlot + the local device shape the HTTP judge sizes its
concurrency from. No scheduler/dispatch here -- the judge is a single-node HTTP service and
agents are assigned to one statically (see test_pipeline.py)."""
from optarena import config
from optarena.agent_bench import judge_scheduler as js
from optarena.agent_bench.judge_scheduler import DeviceSlot, JudgeConfig


def test_device_slot_is_local_and_label():
    local = DeviceSlot("gpu", 1)
    assert local.is_local and local.node is None
    assert local.label == "local:gpu1"
    assert DeviceSlot("cpu", 0).label == "local:cpu0"


def test_local_gpu_count_is_a_nonnegative_int():
    n = js.local_gpu_count()
    assert isinstance(n, int) and n >= 0


def test_judge_config_defaults_from_config(monkeypatch):
    # No configured GPUs and none detected -> a single CPU slot (cpu box default).
    monkeypatch.setattr(js, "local_gpu_count", lambda: 0)
    config.set_override("judge.gpus_per_node", None)
    config.set_override("judge.cpu_slots_per_node", None)
    try:
        cfg = JudgeConfig.from_config()
        assert cfg.gpus_per_node == 0 and cfg.cpu_slots_per_node == 1
    finally:
        config.clear_override("judge.gpus_per_node")
        config.clear_override("judge.cpu_slots_per_node")


def test_judge_config_gpu_box_defaults_no_cpu_slot():
    # GPUs present (configured) -> 0 CPU slots by default (GPU kernels time on the GPU slots).
    config.set_override("judge.gpus_per_node", 4)
    config.set_override("judge.cpu_slots_per_node", None)
    try:
        cfg = JudgeConfig.from_config()
        assert cfg.gpus_per_node == 4 and cfg.cpu_slots_per_node == 0
    finally:
        config.clear_override("judge.gpus_per_node")
        config.clear_override("judge.cpu_slots_per_node")
