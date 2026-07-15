# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``python -m optarena`` entry point -- the same CLI as the ``optarena`` console
script, so a subprocess can spawn a verb through the current interpreter
(``[sys.executable, "-m", "optarena", ...]``) without depending on the console
script being on ``PATH`` (the cluster launcher spawns the judge this way)."""
from optarena.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
