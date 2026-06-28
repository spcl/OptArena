#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Harbor adapter entry point: generate OptArena task directories.

The OptArena -> Harbor logic lives in :mod:`optarena.harbor_adapter` (importable +
unit-tested in the main package); this is the thin CLI Harbor users run, mirroring
the ``algotune`` adapter's generator.

Build the two images once (per hardware target), then generate + run::

    apptainer build optarena-cpu.sif   containers/cpu.def     # agent: toolchain, NO harness
    apptainer build optarena-judge.sif containers/judge.def   # verifier: full harness
    python adapters/optarena/run_adapter.py --output-dir tasks/ --selector all
    python adapters/optarena/run_adapter.py --output-dir tasks/ --selector hpc --group dir
    harbor run -c adapters/optarena/optarena.yaml
"""
import argparse
import sys

from optarena.harbor_adapter import generate, images_for
from optarena.languages import LANG_EXT


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate OptArena Harbor tasks")
    p.add_argument("--output-dir", required=True, help="directory to write the task dirs into")
    p.add_argument("--selector", default="all", help="track / dwarf / kernel or 'all' (default all)")
    p.add_argument("--group", default="kernel", choices=["kernel", "dir"],
                   help="granularity: 'kernel' = one task per kernel (default); "
                   "'dir' = microkernels bundled per directory (microapps stay per-app)")
    p.add_argument("--language", default="c", choices=sorted(LANG_EXT), help="implementation language")
    p.add_argument("--hardware", default="cpu", help="target whose images.<hw> image pair to use (config.yaml)")
    p.add_argument("--agent-image", default=None, help="override the agent image (toolchain, no harness)")
    p.add_argument("--judge-image", default=None, help="override the verifier image (full harness)")
    p.add_argument("--timeout-sec", type=float, default=None, help="verifier timeout (default scales by kernel count)")
    args = p.parse_args(argv)

    agent_image, judge_image = images_for(args.hardware)
    dirs = generate(args.output_dir, selector=args.selector, language=args.language, group=args.group,
                    hardware=args.hardware, agent_image=args.agent_image, judge_image=args.judge_image,
                    timeout_sec=args.timeout_sec)
    print(f"generated {len(dirs)} OptArena tasks (group={args.group}, hardware={args.hardware}) -> {args.output_dir} "
          f"(agent={args.agent_image or agent_image}, verifier={args.judge_image or judge_image})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
