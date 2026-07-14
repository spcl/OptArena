# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unprivileged Apptainer installer for the portable (non-Alps) container path.

Apptainer is the repo's portable image runtime for generic HPC + local use (on Alps the
Slurm-native Container Engine is used instead -- see docs/RUNTIME.md). Apptainer is not
pip-installable (a Go binary); :func:`install_apptainer` runs its official unprivileged
install into a user prefix, exposed as the ``optarena-install-apptainer`` entry point.

The launcher that turns a ``(backend, image, command)`` into an argv is being rebuilt as a
single factory; until then this module carries only the installer.
"""
import os
import subprocess
import sys

#: Apptainer's official unprivileged (no-root) installer.
APPTAINER_INSTALLER = "https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh"


def install_apptainer(prefix="~/.local"):
    """Install Apptainer unprivileged (no sudo) into ``prefix`` via its official
    installer. Returns the subprocess return code.

    The installer is downloaded then piped to ``bash`` over stdin, with ``prefix``
    passed as a real argv element -- NOT interpolated into a ``shell=True`` string
    (which would let a crafted ``prefix`` inject arbitrary commands)."""
    prefix = os.path.expanduser(prefix)
    script = subprocess.run(["curl", "-fsSL", APPTAINER_INSTALLER], check=True, capture_output=True, text=True).stdout
    return subprocess.run(["bash", "-s", "-", prefix], input=script, text=True).returncode


def install_apptainer_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    prefix = argv[0] if argv else "~/.local"
    return install_apptainer(prefix)


if __name__ == "__main__":
    sys.exit(install_apptainer_main())
