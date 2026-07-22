# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thin shim -- the logic now lives in the ``hpcagent_bench`` CLI.

``python scripts/quickstart.py <args>`` is equivalent to ``hpcagent-bench quickstart <args>``
(dispatched to :func:`hpcagent_bench.support.collect.quickstart.quickstart`). Kept so the documented
script path keeps working after the fold into the package CLI.
"""
import sys

from hpcagent_bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["quickstart", *sys.argv[1:]]))
