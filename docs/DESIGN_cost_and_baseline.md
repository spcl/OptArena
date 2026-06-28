# Design — speedup baseline + the cost (token / $) axis

How OptArena measures *performance* and *cost*, and why. Implements the
team-chat consensus: speedup over a **sequential-C** baseline as the safe,
HW-honest performance proxy, and **tokens / $** as the cost axis (frontier-model
users are budget-limited, not time-limited).

## 1. Performance — speedup over sequential C

Every implementation is scored as a **ratio over the same machine's sequential-C
reference**: `r = c_baseline_ns / native_ns`. The C reference is the
NumpyToX-emitted serial scalar C, built `Mode.SINGLE_CORE` (all threading knobs
pinned to 1) — the *consistent fully-serial starting point* all implementations
share, so a 4× means "4× the serial C," comparable across kernel classes
(time-to-completion / GFLOP/s / rounds-per-second all collapse to one ratio).

- **Default everywhere**: `baseline="c"` — `config.yaml` (`service.baseline`), the
  judge `ServiceConfig`, the `runner` (`solve_task`/`run_task`), the `optarena agent`
  CLI, and the OptArena Score (`metric.py`). A kernel that cannot be emitted to C
  (recursive / argmax / not-yet-translatable) **falls back to numpy** for that task,
  recorded honestly in the `baseline` label — never a silent error:
  - `score()` and `measure_baselines()` catch the C build/emit failure and fall back
    to the numpy baseline (the single-shot runner + the service `/baseline` are
    robust);
  - `metric.score_task_fuzzed` additionally pre-probes with
    `scoring.c_reference_available(task)` (cheap, emit-only) so a non-C kernel's `k`
    iterations don't each retry a failing C build.
  - `score()`'s own low-level function default stays `numpy` (so a bare `score(...)`
    call is unchanged); the C default is applied at every high-level entry point.
- **Why not roofline / %-of-peak?** Considered and deferred (chat): it needs HPL/
  STREAM + lscpu/dmidecode rooflines, under-approximates memory-bound kernels
  without a cache model, and does not generalize across every kernel class. A ratio
  over a real serial baseline is HW-honest *and* portable, and is the safest single
  number for a first paper. Roofline normalization remains a drop-in (the geomean
  accepts a normalized `r` unchanged).
- **HW-independence is out of scope.** A score is *for a specific hardware*; we do
  not attempt a cross-architecture-transferable number (complex, hard to generalize,
  confusing — chat consensus).

The OptArena Score is unchanged in shape: `S_i = clamp(geomean_j r(i,j), 1, C_max)`
if solved else 1.0; **OptArena Score = geomean_i S_i** — now a speedup over C.

### 1.1 Measurement rigor — `config.yaml` `measurement.*`

How a single timing is taken is policy, centralized in `config.yaml` `measurement.*`
so the native run and a Harbor run measure identically: `warmup_runs` (untimed,
discarded), `repeat` reps reduced by `aggregation` (`mean`/`median`/`min`),
`pin_threads` (CPU affinity + OMP placement), `n_concurrent_trials: 1` (no co-located
timing). A speedup is kept only if it clears a **geometric-standard-deviation gate**
(`gsd_z`): `S_i / gsd**z > 1`, so a win inside the noise band floors to 1.0. The
Harbor grader applies these; the timing core mirrors them — see
[HANDOFF_measurement_rigor.md](HANDOFF_measurement_rigor.md).

## 2. Cost — tokens, and $ on top

The limiting resource for an agent is *budget*, not wall-clock, so OptArena tracks
the tokens each agent spends and reports **speedup-per-token** (and, with a price
table, **$-to-speedup**).

### 2.1 Capture — pluggable, default = self-report at the score boundary
`optarena/agent_bench/usage.py` defines `TokenUsage(input, output, cached)` with
`total` and `cost_usd(price_table)`. Capture is pluggable:

- **Self-report (built-in).** Each agent reads the token counts its LLM SDK already
  returns — `message.usage` (Anthropic), `prompt_eval_count`/`eval_count` (Ollama) —
  and accumulates them via `Agent.record_usage`. The runner snapshots the cumulative
  total at **each score call** — the boundary we fully control — onto
  `Submission.tokens` and the trajectory. This is the safe in-house path ("we have
  control over score").
- **MITM proxy (future option).** A man-in-the-middle that intercepts *every* LLM
  call — even a closed agent (e.g. Claude Code) talking to its provider — by running
  the agent in a container we set up with the proxy + certificates. It captures
  actual inputs/outputs the agent never exposes. It is a **drop-in for the same
  sink** (`Agent.record_usage` / a per-run accumulator), so nothing downstream
  changes. Not yet built; the seam is in place.

### 2.2 The (tokens, score) trajectory — history per agent call
The repair loop (`runner.solve_task`) records one `CallPoint(round, tokens,
speedup, correct, status)` per agent call — the cumulative **tokens spent so far**
paired with the **score obtained** ("5 tokens before the first run, 15 for the
next, …"). It rides on `RunRow.trajectory` (+ `RunRow.tokens`) and serializes to the
run JSONL. This is the data behind the **performance-vs-tokens** / **performance-vs-$**
plots (the comparison that actually matters: "flagship 80% @ \$400 vs cheaper 75% @
\$15"). Number-of-failed-tries-before-success falls out of the same trajectory.

**Durable history (DB).** `optarena agent --record [--run-id ID]` persists the
trajectory to the results DB — one row per call in the **`calls`** table
(`run_id, ts, benchmark, optimizer, round, tokens, speedup, correct, status,
baseline, …`; schema v2). Unlike `submissions`/`attempts` it is **not** verify-gated:
it records *every* call (passes and failures) because the curve and the
failures-before-success are the point. Queryable across runs for the leaderboard /
plots; the migration is additive (a v1 DB gains `calls` on next open).

### 2.3 Pricing is explicit, not baked in
`TokenUsage.cost_usd(prices)` takes a `{in, out, cache}` $/Mtoken table so a report
is **re-priceable without re-running** — API prices, and especially cache-hit vs
miss policy, change over time and shift the $-number, so we keep the raw token
counts as the durable record and apply prices at report time.

## 3. Roadmap (not in this slice)
- ~~Persist the trajectory to the results DB~~ — **done** (the `calls` table, schema
  v2; `optarena agent --record`).
- **MITM proxy** capture option (container-injected), for closed agents.
- **$ pricing tables** per model + a leaderboard axis (score vs \$).
- ~~Harbor adapter~~ — **done** (`adapters/optarena/`, a Terminal-Bench task
  generator; `optarena/harbor_adapter.py` + `harbor_grade.py`). Uses Harbor's
  **separate verifier environment** (firewall: lean agent image `optarena:cpu`
  without the harness; self-contained verifier image `optarena:judge` from
  `containers/judge.def` with the full harness baked in).
  - **Granularity** (`--group`): `kernel` (one per kernel, default) or `dir`
    (microkernels bundled per directory, reward = geomean of per-kernel `S_i`, gated
    to 1.0 unless all solved; a directory over `max_bundle` emits per-kernel so a flat
    dir like `foundation/` is not one unrunnable task); **microapps always one task
    per app**.
  - **Files, not inlined prompts**: each kernel's reference + C-ABI ship under
    `environment/<kernel>/`, which Harbor uploads to `/app/<kernel>/`; the prompt
    references those container-absolute paths. Submissions cross to the separate
    verifier as `artifacts`, re-materialized at their source path.
  - **Images per hardware** come from `config.yaml` `images.<hw>` (`--hardware`); the
    verifier timeout scales by kernel count. No oracle solution is shipped (it would
    need the harness in the agent image — firewall); gradeability is covered by tests.
  - Remaining: **build + publish** the images on real infra (defs not build-verified
    in the sandbox); a Docker `judge.Dockerfile` variant; a self-contained oracle
    (pre-emitted C under `solution/`).
