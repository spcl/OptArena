# Archived / removed components

Components cut from the repo during the 2026-07-13 cleanup, recorded here so they
can be found and restored if needed. Nothing here was in the live import graph —
each removal is proven dead-by-grep in the cleanup notes.

## `optarena/hardware_info/theoretical/` — roofline hardware table (ARCHIVED)

- **Files:** `cpu_info.yaml` (peak FLOP/BW table), `cpu_gpu_info.py`,
  `memory_info.py` (dmidecode/CPU probes), plus the `hardware_info/theoretical/*.yaml`
  entry in `setup.py` `package_data`.
- **Why removed:** fully orphaned. No module imports `hardware_info`,
  `cpu_gpu_info`, `memory_info`, or `get_cpu_info` (only reference was the
  `setup.py` package glob). The `OPTARENA_DMIDECODE_DUMP` env var was consumed
  only by `memory_info.py`, so it goes with the tree.
- **Restore:** the tree is preserved on branch **`archive/hardware-info`**
  (pushed to origin), pointing at the pre-removal commit. `git checkout
  archive/hardware-info -- optarena/hardware_info` brings it back, then re-add the
  `package_data` glob. It was intended for a roofline overlay that was never
  wired into scoring; revisit there if roofline lands.

## Deleted outright (not archived — trivially regenerated or superseded)

- `optarena/taxonomy/dwarfs.yaml` — never loaded; the live closed set is the
  hard-coded `SUPPORTED_DWARFS` frozenset in `optarena/spec.py`.
- `optarena/schemas/bench_spec.schema.yaml` — never machine-enforced; the live
  validator is `KNOWN_MANIFEST_KEYS` + `BenchSpec.from_yaml` in `optarena/spec.py`.
  The `# Schema: ...` pointer comment was stripped from all ~343 manifests.
- Root `Dockerfile` — legacy demo image (`FROM python:3`); superseded by the
  real per-hardware `containers/{cpu,nvidia,amd}.Dockerfile`.
