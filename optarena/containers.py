# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Container launch factory + the unprivileged Apptainer installer.

One factory (:func:`local_run_command`) turns a ``(backend, image, command)`` into a launch
argv. Only apptainer + podman are supported -- both are what CSCS launches, and both consume
the ONE universal OCI image (apptainer builds a SIF from it; podman runs the OCI tag). The
per-backend flag SPELLINGS live in the language-neutral ``container_backends.txt`` (this
directory), read here by Python and by ``scripts/run_agent_in_container.sh`` in pure bash --
one source of truth for both the Python callers and the python-less HPC login host.

Harbor is an orchestrator, not a wrapper; :func:`harbor_env_for` only supplies its provider
name (apptainer -> singularity).

Apptainer itself is a Go binary (not pip-installable); :func:`install_apptainer` runs its
official unprivileged install into a user prefix, exposed as the ``optarena-install-apptainer``
entry point.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from optarena import config

#: Apptainer's official unprivileged (no-root) installer.
APPTAINER_INSTALLER = "https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh"

#: The single-source spelling file, read by BOTH this module and the bash launcher.
BACKENDS_PATH = pathlib.Path(__file__).parent / "container_backends.txt"

#: The selectable backends -- apptainer + podman, both exec-wrapper rows in the spelling file.
KNOWN_BACKENDS = ("apptainer", "podman")


@dataclass(frozen=True)
class WrapperSpelling:
    """How one exec-wrapper backend spells its launch flags (one row of the file)."""
    name: str
    verb: Tuple[str, ...]  # ("exec",) | ("run", "--rm", "--network", "host") | ("run", "--rm")
    bind_flag: str  # "--bind" | "-v"
    workdir_flag: str  # "--pwd" | "-w"
    env_flag: str  # "--env" | "-e"
    gpu: Mapping[str, Tuple[str, ...]]  # {"nvidia": (...), "amd": (...)}; a cpu run adds nothing
    image_form: str  # "sif" | "tag"
    image_default: str  # "optarena-{hw}.sif" | "optarena:{hw}"
    harbor_env: str  # "singularity" | "docker" | "" (empty = not a Harbor backend)


def load_backends(path: pathlib.Path = BACKENDS_PATH) -> Tuple[dict, Tuple[str, ...]]:
    """Parse the spelling file into ``({backend: WrapperSpelling}, passthrough_env)``.

    Both the Python fold and the bash fold read this one file, so the launch argv is
    byte-identical across the language boundary."""
    rows: dict = {}
    passthrough: Tuple[str, ...] = ()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        head, _, field = key.strip().partition(".")
        if head == "global":
            if field == "passthrough":
                passthrough = tuple(value.split())
            continue
        rows.setdefault(head, {})[field] = value
    spellings = {
        name:
        WrapperSpelling(name=name,
                        verb=tuple(f["verb"].split()),
                        bind_flag=f["bind"].strip(),
                        workdir_flag=f["workdir"].strip(),
                        env_flag=f["env"].strip(),
                        gpu={
                            "nvidia": tuple(f.get("gpu.nvidia", "").split()),
                            "amd": tuple(f.get("gpu.amd", "").split()),
                        },
                        image_form=f["image_form"].strip(),
                        image_default=f["image_default"].strip(),
                        harbor_env=f.get("harbor_env", "").strip())
        for name, f in rows.items()
    }
    return spellings, passthrough


SPELLINGS, PASSTHROUGH_ENV = load_backends()


def resolve_backend(explicit: Optional[str] = None) -> str:
    """The active container backend: ``explicit`` arg > ``$OPTARENA_RUNTIME_BACKEND`` >
    ``config.get("runtime.backend")`` > ``apptainer``.

    Note: the legacy bash-only ``$OPTARENA_CONTAINER_RUNTIME`` is DELIBERATELY not read
    here -- the shell launcher still honors it locally, but wiring it into the Python
    path would make a Harbor run crash whenever a user had set it for a local bash run.
    Both paths share the one canonical ``$OPTARENA_RUNTIME_BACKEND``."""
    backend = (explicit or os.environ.get("OPTARENA_RUNTIME_BACKEND") or config.get("runtime.backend", "apptainer")
               or "apptainer").strip()
    if backend not in KNOWN_BACKENDS:
        raise ValueError(f"unknown container backend {backend!r}; known: {list(KNOWN_BACKENDS)}")
    return backend


def default_image(backend: str, hardware: str = "cpu", repo_root: Optional[str] = None) -> str:
    """The image reference for ``backend`` on ``hardware`` -- an ``$OPTARENA_SIF`` /
    ``$OPTARENA_DOCKER_IMAGE`` override, else the file's default (a sif path under
    ``repo_root``, or an ``optarena:<hw>`` tag)."""
    spelling = SPELLINGS[backend]
    if spelling.image_form == "sif":
        override = os.environ.get("OPTARENA_SIF")
        if override:
            return override
        name = spelling.image_default.format(hw=hardware)
        return os.path.join(repo_root, name) if repo_root else name
    return os.environ.get("OPTARENA_DOCKER_IMAGE") or spelling.image_default.format(hw=hardware)


def collect_env(hardware: str) -> List[Tuple[str, str]]:
    """The ``(key, value)`` env pairs to forward into the image, in a PINNED order so the
    bash fold matches byte-for-byte: ``OPTARENA_IMAGE=<hw>`` first, then
    :data:`PASSTHROUGH_ENV` (present, in file order), then every other ``OPTARENA_*`` var
    sorted (Python's str sort == ``LC_ALL=C sort``). Reads only the environment -- there is no
    caller-supplied extra, because the bash fold has no such channel and any divergence would
    silently break the byte-for-byte parity."""
    pairs: List[Tuple[str, str]] = [("OPTARENA_IMAGE", hardware)]
    seen = {"OPTARENA_IMAGE"}
    for key in PASSTHROUGH_ENV:
        value = os.environ.get(key)
        if value and key not in seen:
            pairs.append((key, value))
            seen.add(key)
    for key in sorted(k for k in os.environ if k.startswith("OPTARENA_") and k not in seen):
        value = os.environ.get(key)
        if value:
            pairs.append((key, value))
            seen.add(key)
    # Invariant (container_backends.txt): the fold is newline-delimited on the bash side, so a
    # value with a newline would split into extra argv tokens there while staying one token here
    # -- a silent parity break. Fail loud rather than emit a corrupt launch.
    for key, value in pairs:
        if "\n" in value:
            raise ValueError(f"env {key!r} contains a newline; the launch fold is newline-delimited "
                             f"and cannot forward it (container_backends.txt token-list invariant)")
    return pairs


def local_run_command(inner: Sequence[str],
                      *,
                      backend: Optional[str] = None,
                      hardware: str = "cpu",
                      image: Optional[str] = None,
                      repo_root: Optional[str] = None) -> List[str]:
    """THE factory: the full launch argv for running ``inner`` inside the image under an
    exec-wrapper backend -- ``prefix + [image] + inner`` in the fixed fold order the bash
    launcher mirrors. ``backend`` defaults to :func:`resolve_backend` (apptainer | podman)."""
    chosen = resolve_backend(backend)
    spelling = SPELLINGS[chosen]
    repo = repo_root or os.getcwd()
    argv: List[str] = [chosen, *spelling.verb, *spelling.gpu.get(hardware, ())]
    for key, value in collect_env(hardware):
        argv += [spelling.env_flag, f"{key}={value}"]
    argv += [spelling.bind_flag, f"{repo}:{repo}", spelling.workdir_flag, repo]
    argv.append(image or default_image(chosen, hardware, repo))
    argv += list(inner)
    return argv


def harbor_env_for(backend: Optional[str] = None) -> str:
    """Harbor's ``--env`` provider name for the resolved backend (``apptainer -> singularity``).
    Raises for ``podman`` (Harbor drives singularity + docker only), so the caller never emits
    an invalid provider -- a podman run is launched directly, not through Harbor."""
    chosen = resolve_backend(backend)
    name = SPELLINGS[chosen].harbor_env
    if not name:
        raise ValueError(f"{chosen!r} is not a Harbor backend (Harbor provides singularity + docker); "
                         "run it directly via local_run_command / scripts/run_agent_in_container.sh")
    return name


def install_apptainer(prefix="~/.local", attempts=4):
    """Install Apptainer unprivileged (no sudo) into ``prefix`` via its official
    installer. Returns the subprocess return code.

    The installer is downloaded then piped to ``bash`` over stdin, with ``prefix``
    passed as a real argv element -- NOT interpolated into a ``shell=True`` string
    (which would let a crafted ``prefix`` inject arbitrary commands).

    Retried with backoff (as ``pip_retry`` does for the CI pip installs) because BOTH
    fetches are live-network: the installer itself, and the EPEL package listing the
    installer scrapes to resolve the latest apptainer RPM. That listing is served by the
    ``download.fedoraproject.org`` REDIRECTOR, so a single bad mirror fails the install
    outright. Upstream's own retry loop cannot absorb that -- it NEVER sleeps between
    attempts, so a momentarily unreachable mirror burns all of its retries in under a second
    (seen in CI: five attempts, 0.80 s total, against a listing that resolves in 0.43 s when
    healthy). Retrying the whole script in a FRESH process is what actually helps: the
    installer caches the fetched listing in a shell variable and skips the re-fetch when
    it is non-empty, so only a new process re-queries the redirector and can land on a
    different mirror.

    Any partial tree a failed attempt left behind is removed before the next one. This is
    what makes the retry work at all: the installer hard-refuses when its own
    ``<prefix>/<arch>`` already exists (``fatal "$DEST/$ARCH is not empty"``, and it has no
    force flag), and a mirror that dies midway has already unpacked into it -- so without
    the clean, every retry fails INSTANTLY on that check instead of re-fetching, and the
    real error is buried under "is not empty" (seen in CI: a bad mirror lost
    ``fakeroot-libs``, then three retries reported only the leftover directory).
    :func:`clean_partial_install` removes only paths this call created."""
    prefix = os.path.expanduser(prefix)
    preexisting = set(os.listdir(prefix)) if os.path.isdir(prefix) else set()
    returncode = 1
    for attempt in range(1, attempts + 1):
        try:
            script = subprocess.run(["curl", "-fsSL", APPTAINER_INSTALLER], check=True, capture_output=True,
                                    text=True).stdout
            returncode = subprocess.run(["bash", "-s", "-", prefix], input=script, text=True).returncode
            if returncode == 0:
                return 0
        except subprocess.CalledProcessError as exc:
            returncode = exc.returncode
        if attempt < attempts:
            clean_partial_install(prefix, preexisting)
            delay = 5 * attempt
            print(f"apptainer install attempt {attempt}/{attempts} failed (rc={returncode}); retrying in {delay}s",
                  file=sys.stderr)
            time.sleep(delay)
    return returncode


def clean_partial_install(prefix: str, preexisting: Sequence[str]) -> None:
    """Remove what a failed :func:`install_apptainer` attempt left in ``prefix`` -- and ONLY that.

    ``preexisting`` is the prefix's entries from before the first attempt; anything named there is
    left alone. Scoping it this way is the whole point rather than a nicety: ``prefix`` defaults to
    ``~/.local`` and is caller-supplied, so a blanket wipe of it would delete a user's unrelated
    installs. Only the names the installer itself added (its ``<arch>`` tree and ``bin`` shims) are
    candidates."""
    if not os.path.isdir(prefix):
        return
    for name in os.listdir(prefix):
        if name in preexisting:
            continue
        path = os.path.join(prefix, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def install_apptainer_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    prefix = argv[0] if argv else "~/.local"
    return install_apptainer(prefix)


if __name__ == "__main__":
    sys.exit(install_apptainer_main())
