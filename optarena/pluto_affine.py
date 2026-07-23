# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Affine-index detection for a ``#pragma scop`` Pluto input.

Pluto is a polyhedral (affine) optimizer: an array access whose subscript index is non-affine is
outside its model, and ``polycc`` may silently MISCOMPILE such a scop rather than reject it. The
detector here scans the scop's array subscripts and returns the first non-affine index pattern, so a
caller can deem Pluto inapplicable (a clean skip) instead of trusting a transform it cannot soundly
make.

This lives in the package (not under ``tests/``) so both the numerical oracle AND external consumers
(e.g. the nest-forge arena's Pluto lane) import the SAME detector rather than reimplementing it.
``tests.numerical_oracle`` re-exports it under its historical private name.
"""
from __future__ import annotations

import re
from typing import Optional


def scop_nonaffine_reason(scop_c: str) -> Optional[str]:
    """Return the first non-affine array-subscript pattern in a ``#pragma scop`` body, or ``None`` when
    every subscript index is affine.

    The patterns, outside Pluto's polyhedral model, are ``indirection`` (``b[ip[i]]``), ``modulo``
    (``a[i % k]``) and ``integer-division`` (``a[i / k]``). Only the index INSIDE ``[...]`` matters --
    value-side ``/`` and ``%`` are ignored, so an affine program Pluto merely miscompiles stays a tracked
    FAIL, not a skip. When no ``#pragma scop``/``#pragma endscop`` pair is present the whole string is
    scanned (an already-extracted scop body)."""
    m = re.search(r"#pragma scop(.*?)#pragma endscop", scop_c, re.S)
    body = m.group(1) if m else scop_c
    i, n = 0, len(body)
    while i < n:
        if body[i] != "[":
            i += 1
            continue
        depth, j, inner = 1, i + 1, []
        while j < n and depth:
            c = body[j]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            inner.append(c)
            j += 1
        sub = "".join(inner)
        if "[" in sub:
            return "indirection"
        if "%" in sub:
            return "modulo"
        if "/" in sub:
            return "integer-division"
        i = j + 1
    return None
