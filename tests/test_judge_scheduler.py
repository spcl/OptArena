# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Judge device scheduler: slot planning, dynamic dispatch, GPU pinning, config."""
import threading
import time

from optarena.agent_bench import native_call
from optarena.agent_bench.judge_scheduler import (AgentPoolConfig, DeviceSlot, JudgeConfig, JudgeScheduler,
                                                  TwoStageScheduler, srun_wrap)


def test_slots_gpu_then_cpu_per_node():
    cfg = JudgeConfig(gpus_per_node=2, cpu_slots_per_node=1)
    slots = cfg.slots()
    assert [s.label for s in slots] == ["local:gpu0", "local:gpu1", "local:cpu0"]
    assert cfg.nodes == 1


def test_slots_multinode():
    cfg = JudgeConfig(gpus_per_node=4, cpu_slots_per_node=0, nodelist=("nid001", "nid002"))
    slots = cfg.slots()
    assert cfg.nodes == 2
    assert len(slots) == 8
    assert slots[0] == DeviceSlot("gpu", 0, "nid001")
    assert slots[4] == DeviceSlot("gpu", 0, "nid002")
    assert all(not s.is_local for s in slots)


def test_slots_fallback_to_one_cpu():
    assert JudgeConfig(gpus_per_node=0, cpu_slots_per_node=0).slots() == [DeviceSlot("cpu", 0, None)]


def test_from_config_defaults(monkeypatch):
    # No GPUs visible in a CPU test env -> 0 gpu slots, 1 cpu slot.
    monkeypatch.setattr("optarena.agent_bench.judge_scheduler.local_gpu_count", lambda: 0)
    monkeypatch.delenv("OPTARENA_JUDGE_GPUS_PER_NODE", raising=False)
    monkeypatch.delenv("OPTARENA_JUDGE_CPU_SLOTS_PER_NODE", raising=False)
    cfg = JudgeConfig.from_config()
    assert cfg.gpus_per_node == 0 and cfg.cpu_slots_per_node == 1


def test_run_processes_all_in_order():
    sched = JudgeScheduler([DeviceSlot("cpu", 0), DeviceSlot("cpu", 1)])
    out = sched.run(list(range(20)), lambda it, slot: it * it)
    assert [v for _s, v in out] == [i * i for i in range(20)]
    assert all(s == "ok" for s, _v in out)


def test_error_is_captured_not_fatal():

    def run_item(it, slot):
        if it == 3:
            raise ValueError("boom")
        return it

    out = JudgeScheduler([DeviceSlot("cpu", 0)]).run(list(range(5)), run_item)
    assert out[3][0] == "err" and isinstance(out[3][1], ValueError)
    assert [v for s, v in out if s == "ok"] == [0, 1, 2, 4]


def test_gpu_slot_pins_thread_local():
    # A gpu slot exposes its index via native_call.assigned_device() inside run_item;
    # a cpu slot leaves it None. (No real GPU needed -- we only read the thread-local.)
    seen = {}

    def run_item(it, slot):
        seen[it] = native_call.assigned_device()
        return it

    slots = [DeviceSlot("gpu", 0), DeviceSlot("gpu", 1), DeviceSlot("cpu", 0)]
    JudgeScheduler(slots).run([0, 1, 2, 3, 4, 5], run_item)
    # every item saw a valid pin (a gpu index int, or None for a cpu slot); and the
    # pin is cleared after (no leak into the main thread).
    assert set(seen.values()) <= {0, 1, None}
    assert native_call.assigned_device() is None


def test_dynamic_work_stealing():
    # 2 slots, 6 items; one slot's items are slow. A static split would make the
    # fast slot finish 3 quick items then idle; work-stealing keeps it pulling.
    order_lock = threading.Lock()
    finish_order = []

    def run_item(it, slot):
        time.sleep(0.02 if it == 0 else 0.001)  # item 0 is the slow one
        with order_lock:
            finish_order.append(it)
        return it

    JudgeScheduler([DeviceSlot("cpu", 0), DeviceSlot("cpu", 1)]).run(list(range(6)), run_item)
    # The slow item 0 finishes late while the other slot drains the rest -> item 0
    # is NOT among the first finishers (proves the fast slot kept stealing work).
    assert 0 in finish_order[-2:]
    assert len(finish_order) == 6


def test_srun_wrap_remote_vs_local():
    launcher = ("srun", "--nodelist", "{node}", "--gpus", "1", "-n", "1")
    local = srun_wrap(DeviceSlot("gpu", 0, None), ["bench", "in", "out"], launcher)
    remote = srun_wrap(DeviceSlot("gpu", 2, "nid007"), ["bench", "in", "out"], launcher)
    assert local == ["bench", "in", "out"]
    assert remote == ["srun", "--nodelist", "nid007", "--gpus", "1", "-n", "1", "bench", "in", "out"]


def test_native_call_assigned_device_roundtrip():
    assert native_call.assigned_device() is None
    native_call.set_assigned_device(3)
    assert native_call.assigned_device() == 3
    native_call.set_assigned_device(None)
    assert native_call.assigned_device() is None


# ---- Agent pool + two-stage (think -> grade) pipeline ----------------------


def test_agent_slots_per_node_and_multinode():
    cfg = AgentPoolConfig(workers_per_node=3, nodelist=("nid001", "nid002"))
    slots = cfg.slots()
    assert cfg.nodes == 2 and len(slots) == 6
    assert slots[0] == DeviceSlot("agent", 0, "nid001")
    assert slots[3] == DeviceSlot("agent", 0, "nid002")
    assert all(s.kind == "agent" for s in slots)


def test_agent_pool_from_config_defaults(monkeypatch):
    monkeypatch.delenv("OPTARENA_AGENT_WORKERS_PER_NODE", raising=False)
    monkeypatch.delenv("OPTARENA_AGENT_NODELIST", raising=False)
    monkeypatch.delenv("OPTARENA_AGENT_NODES_EXPANDED", raising=False)
    cfg = AgentPoolConfig.from_config()
    assert cfg.workers_per_node == 1 and cfg.nodes == 1
    assert cfg.slots() == [DeviceSlot("agent", 0, None)]


def test_two_stage_think_then_grade_in_order():
    # think doubles, grade adds one -> value = it*2 + 1, in input order.
    sched = TwoStageScheduler([DeviceSlot("agent", 0), DeviceSlot("agent", 1)],
                              [DeviceSlot("cpu", 0), DeviceSlot("cpu", 1)])
    out = sched.run(list(range(20)), think=lambda it, s: it * 2, grade=lambda cand, it, s: cand + 1)
    assert [v for st, v in out] == [i * 2 + 1 for i in range(20)]
    assert all(st == "ok" for st, _v in out)


def test_two_stage_routes_to_the_right_pool():
    kinds = {}

    def think(it, slot):
        kinds.setdefault(it, {})["think"] = slot.kind
        return it

    def grade(cand, it, slot):
        kinds[it]["grade"] = slot.kind
        return cand

    TwoStageScheduler([DeviceSlot("agent", 0)], [DeviceSlot("gpu", 0)]).run([0, 1, 2], think, grade)
    assert all(v == {"think": "agent", "grade": "gpu"} for v in kinds.values())


def test_two_stage_think_error_skips_grade():
    graded = []

    def think(it, slot):
        if it == 2:
            raise ValueError("think boom")
        return it

    def grade(cand, it, slot):
        graded.append(it)
        return cand

    out = TwoStageScheduler([DeviceSlot("agent", 0)], [DeviceSlot("cpu", 0)]).run(list(range(5)), think, grade)
    assert out[2][0] == "err" and isinstance(out[2][1], ValueError)
    assert 2 not in graded  # a failed think is never graded
    assert sorted(graded) == [0, 1, 3, 4]
    assert [v for st, v in out if st == "ok"] == [0, 1, 3, 4]


def test_two_stage_grade_error_captured():

    def grade(cand, it, slot):
        if it == 1:
            raise RuntimeError("grade boom")
        return cand

    out = TwoStageScheduler([DeviceSlot("agent", 0)], [DeviceSlot("cpu", 0)]).run(list(range(3)),
                                                                                  think=lambda it, s: it,
                                                                                  grade=grade)
    assert out[1][0] == "err" and isinstance(out[1][1], RuntimeError)
    assert [v for st, v in out if st == "ok"] == [0, 2]


def test_two_stage_pipelines_without_a_barrier():
    # ONE agent slot -> thinks are serial; grade is instant. If the pipeline had a
    # barrier (all think, then all grade) the first grade would see ALL thinks done.
    # Pipelined, the first grade fires while later items are still thinking.
    lock = threading.Lock()
    thinks_done = [0]
    at_first_grade = []

    def think(it, slot):
        time.sleep(0.01)
        with lock:
            thinks_done[0] += 1
        return it

    def grade(cand, it, slot):
        with lock:
            if not at_first_grade:
                at_first_grade.append(thinks_done[0])
        return cand

    TwoStageScheduler([DeviceSlot("agent", 0)], [DeviceSlot("cpu", 0)]).run(list(range(6)), think, grade)
    assert at_first_grade and at_first_grade[0] < 6  # grading started before all thinks finished


def test_two_stage_empty_items():
    assert TwoStageScheduler([DeviceSlot("agent", 0)], [DeviceSlot("cpu", 0)]).run([],
                                                                                   think=lambda it, s: it,
                                                                                   grade=lambda c, it, s: c) == []
