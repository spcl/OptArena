# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch drivers over the kernel registry: sweep (framework-baseline sweeps into optarena.db),
quickstart (tiny demo sweep), pluto_survey (affine-backend survey) -- dispatched by the optarena CLI,
which defers importing these (and their heavy per-framework imports) until a subcommand runs."""
