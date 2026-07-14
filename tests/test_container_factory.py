# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the container-launch factory (optarena/containers.py) -- argv assembly,
backend resolution (apptainer | podman), and the Harbor provider name. Pure argv assertions:
no real container, GPU, or LLM, so this runs on any CI runner.

The bash<->Python golden parity test lives in test_container_launch_parity.py."""
import os

import pytest

from optarena import containers


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch):
    """Drop every ambient container/runtime var so a developer's shell cannot skew the
    argv assertions; each test sets only what it needs."""
    for key in list(os.environ):
        if key.startswith("OPTARENA_") or key in ("OLLAMA_HOST", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    yield


def test_load_backends_lists_the_two_exec_wrappers():
    spellings, passthrough = containers.load_backends()
    assert set(spellings) == {"apptainer", "podman"}
    assert "ANTHROPIC_API_KEY" in passthrough
    assert spellings["apptainer"].verb == ("exec", )
    assert spellings["podman"].verb == ("run", "--rm", "--network", "host")


def test_resolve_backend_precedence(monkeypatch):
    assert containers.resolve_backend("podman") == "podman"  # explicit wins
    monkeypatch.setenv("OPTARENA_RUNTIME_BACKEND", "podman")
    assert containers.resolve_backend() == "podman"  # canonical env next
    monkeypatch.delenv("OPTARENA_RUNTIME_BACKEND")
    assert containers.resolve_backend() == "apptainer"  # config/code default


def test_resolve_backend_ignores_the_legacy_bash_var(monkeypatch):
    # The decouple fix: $OPTARENA_CONTAINER_RUNTIME is the shell launcher's own knob and
    # must NOT steer the Python path. Only $OPTARENA_RUNTIME_BACKEND is shared.
    monkeypatch.setenv("OPTARENA_CONTAINER_RUNTIME", "podman")
    assert containers.resolve_backend() == "apptainer"


def test_resolve_backend_rejects_unknown():
    for dropped in ("singularity", "docker", "udocker", "ce"):
        with pytest.raises(ValueError):
            containers.resolve_backend(dropped)


def test_local_run_command_apptainer_cpu():
    argv = containers.local_run_command(["python", "-m", "optarena.cli", "agent"],
                                        backend="apptainer",
                                        hardware="cpu",
                                        repo_root="/repo")
    assert argv == [
        "apptainer", "exec", "--env", "OPTARENA_IMAGE=cpu", "--bind", "/repo:/repo", "--pwd", "/repo",
        "/repo/optarena-cpu.sif", "python", "-m", "optarena.cli", "agent"
    ]


def test_local_run_command_podman_nvidia_gpu_tokens():
    argv = containers.local_run_command(["run"], backend="podman", hardware="nvidia", repo_root="/r")
    # podman run --rm --network host --device nvidia.com/gpu=all ...
    assert argv[:5] == ["podman", "run", "--rm", "--network", "host"]
    assert "--device" in argv and "nvidia.com/gpu=all" in argv
    assert argv[-2:] == ["optarena:nvidia", "run"]


def test_local_run_command_podman_amd_gpu_tokens():
    argv = containers.local_run_command(["x"], backend="podman", hardware="amd", repo_root="/r")
    assert "/dev/kfd" in argv and "--group-add" in argv and "keep-groups" in argv


def test_local_run_command_rejects_dropped_backend():
    for dropped in ("docker", "udocker", "ce"):
        with pytest.raises(ValueError):
            containers.local_run_command(["x"], backend=dropped)


def test_default_image_sif_tag_and_overrides(monkeypatch):
    assert containers.default_image("apptainer", "cpu", repo_root="/r") == "/r/optarena-cpu.sif"
    assert containers.default_image("podman", "nvidia") == "optarena:nvidia"
    monkeypatch.setenv("OPTARENA_SIF", "/scratch/my.sif")
    assert containers.default_image("apptainer", "cpu", repo_root="/r") == "/scratch/my.sif"
    monkeypatch.setenv("OPTARENA_DOCKER_IMAGE", "reg/img:tag")
    assert containers.default_image("podman", "cpu") == "reg/img:tag"


def test_collect_env_order_is_pinned(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")  # a passthrough (non-OPTARENA) var
    monkeypatch.setenv("OPTARENA_ZED", "z")  # dynamic OPTARENA_*, sorts last
    monkeypatch.setenv("OPTARENA_ABC", "a")  # dynamic OPTARENA_*, sorts before ZED
    pairs = containers.collect_env("cpu")
    assert pairs[0] == ("OPTARENA_IMAGE", "cpu")  # image first
    assert ("ANTHROPIC_API_KEY", "sk") in pairs
    keys = [k for k, _ in pairs]
    assert keys.index("OPTARENA_ABC") < keys.index("OPTARENA_ZED")  # sorted
    assert keys.count("OPTARENA_IMAGE") == 1  # no duplicate


def test_collect_env_rejects_a_newline_value(monkeypatch):
    monkeypatch.setenv("OPTARENA_BAD", "line1\nline2")
    with pytest.raises(ValueError):
        containers.collect_env("cpu")


def test_harbor_env_for_maps_and_raises():
    assert containers.harbor_env_for("apptainer") == "singularity"
    with pytest.raises(ValueError):
        containers.harbor_env_for("podman")  # podman is launched directly, not via Harbor
