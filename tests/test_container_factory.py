# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the container-launch factory (optarena/containers.py) -- argv assembly,
backend resolution (apptainer | podman), and the Harbor provider name. Pure argv assertions:
no real container, GPU, or LLM, so this runs on any CI runner.

The bash<->Python golden parity test lives in test_container_launch_parity.py."""
import os
import subprocess

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


# --------------------------------------------------------------------------- #
# install_apptainer retry. Both fetches are live-network (the installer, and the EPEL
# listing it scrapes through the fedoraproject REDIRECTOR), so a single bad mirror used to
# fail the whole install -- it reds the CI container job. subprocess/sleep are stubbed, so
# these stay pure-unit: no network, no real apptainer.
# --------------------------------------------------------------------------- #


def _stub_installer(monkeypatch, bash_returncodes, curl_error=None):
    """Stub curl+bash. ``bash_returncodes`` is consumed one per install attempt; ``curl_error``
    (a returncode) instead makes the FIRST curl raise, as a failed download does. Records the
    executable of every call plus every backoff delay slept.

    ``containers.subprocess`` IS the one global subprocess module object, so patching ``run`` on it
    hijacks EVERY caller in the process, not just install_apptainer's. Anything else that shells out
    while the patch is live (a pytest plugin, coverage) would land in here and eat a
    ``bash_returncodes`` entry -- an intermittent failure with no connection to the code under test.
    So only the two argv this stub models are intercepted; everything else is delegated to the real
    subprocess.run."""
    calls, sleeps, pending = [], [], list(bash_returncodes)
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        if not argv or argv[0] not in ("curl", "bash"):
            return real_run(argv, **kwargs)  # not ours -- never consume a queued returncode
        calls.append(argv[0])
        if argv[0] == "curl":
            failed = curl_error is not None and calls.count("curl") == 1
            returncode = curl_error if failed else 0
            stdout = "" if failed else "#!/bin/bash\ntrue\n"
        else:
            returncode, stdout = pending.pop(0), ""
        # Honour subprocess.run's REAL contract -- check=True is what turns a nonzero rc into
        # CalledProcessError. Raising straight from the stub flag instead would leave the
        # ``check=True`` at the curl call site untested, and dropping it there is a silent
        # false-success: a failed curl returns nonzero with EMPTY stdout, the empty script pipes
        # to bash, bash exits 0, and the install reports success having installed nothing.
        if kwargs.get("check") and returncode != 0:
            raise subprocess.CalledProcessError(returncode, argv)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout)

    monkeypatch.setattr(containers.subprocess, "run", fake_run)
    monkeypatch.setattr(containers.time, "sleep", lambda s: sleeps.append(s))
    return calls, sleeps


def test_install_apptainer_retries_a_transient_mirror_failure(monkeypatch):
    """A mirror blip is retried in a FRESH process -- the point of retrying at OUR level.

    Upstream's own loop cannot recover from this: it caches the fetched listing in a shell
    variable and skips the re-fetch while that is non-empty, so only a new process re-queries
    the redirector and can land on a different mirror."""
    calls, sleeps = _stub_installer(monkeypatch, bash_returncodes=[2, 2, 0])
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=4) == 0
    assert calls.count("bash") == 3, "each attempt must re-run the installer in a fresh process"
    assert sleeps == [5, 10], "backoff must grow, and must NOT sleep after the attempt that succeeded"


def test_install_apptainer_gives_up_and_reports_the_installer_returncode(monkeypatch):
    """Exhausting the attempts still surfaces the real failure -- never a false success."""
    calls, sleeps = _stub_installer(monkeypatch, bash_returncodes=[2, 2, 2])
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=3) == 2
    assert calls.count("bash") == 3
    assert sleeps == [5, 10], "no trailing sleep after the final attempt"


def test_install_apptainer_retries_a_failed_installer_download(monkeypatch):
    """The installer download is live-network too, so a curl failure retries rather than raising."""
    calls, _ = _stub_installer(monkeypatch, bash_returncodes=[0], curl_error=6)
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=3) == 0
    assert calls.count("curl") == 2, "the failed download must be re-fetched, not raised to the caller"


def test_install_apptainer_succeeds_first_try_without_sleeping(monkeypatch):
    """The happy path must not pay any backoff (guards against an off-by-one in the loop)."""
    calls, sleeps = _stub_installer(monkeypatch, bash_returncodes=[0])
    assert containers.install_apptainer("/tmp/apptainer-prefix") == 0
    assert calls.count("bash") == 1
    assert sleeps == []
