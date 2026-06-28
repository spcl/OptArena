# Handoff — measurement rigor (verification-system side)

The Harbor adapter (`harbor_adapter.py`, `harbor_grade.py`) is done and reads all
measurement policy from `config.yaml` `measurement.*`. Three knobs live in the timing
core (`scoring.py` / `metric.py`), which the adapter does not own — please wire them
to the same config keys so the native and Harbor paths measure identically.

`config.yaml` `measurement:` keys:

| key | meaning | adapter status | needs |
|-----|---------|----------------|-------|
| `baseline` | speedup denominator (`c`) | used | — |
| `repeat` | timed reps per measurement | passed to `score_task_fuzzed` | — |
| `c_max` | clamp ceiling | used | — |
| `gsd_z` | dispersion-gate z | applied in `harbor_grade` from per-iteration timings | mirror in `metric.aggregate` so native `S_i` is gated too (else native != Harbor) |
| `warmup_runs` | untimed runs before timing | inert | add a warmup loop in `scoring.py` before the timed reps |
| `aggregation` | reduce reps: `mean`/`median`/`min` | inert (core keeps `min`) | honor in `scoring.py` (default `mean`) |
| `pin_threads` | affinity + OMP placement | `harbor_grade.pin_threads()` | reuse for the native CLI sweep |
| `n_concurrent_trials` | no co-located timing | set in `optarena.yaml`; native sweep should honor it | — |

Parity note: until `gsd_z`/`aggregation`/`warmup_runs` land in the core, the Harbor
reward is the gated/clamped `S_i` while native `S_i` is ungated — same correctness,
slightly different headline. `harbor_grade.grade()` already returns `gsd` and the
per-iteration `native_ns`/`baseline_ns`, so the gate can move into `metric` verbatim.
