# PyTorch to NumPy Translator

Toolchain that converts PyTorch KernelBench kernels into minimal NumPy.

## Layout

- `level1/`, `level2/`, `level3/` hold the source PyTorch inputs.
- `../benchmarks/ml/KernelBench/level*/` holds generated benchmark outputs.
  Each translated module is written as `<source>_numpy.py`, with a sibling
  `<source>.yaml` manifest containing benchmark parameters and input arrays.
- `src/` contains translator code.
- `test/` contains parity tests and harness code.
- `skills/` contains the VS Code skill metadata for this workflow.

## Conventions

- Keep source inputs in the `level*` folders unchanged unless you are
  intentionally fixing the corpus.
- Write translated outputs as standalone NumPy modules under
  `optarena/benchmarks/ml/KernelBench/level*/`.
- Keep source-only sizing globals and `get_inputs`/`get_init_inputs` metadata
  out of generated NumPy modules; that benchmark metadata belongs in YAML.

`CONTRIBUTOR_GUIDE.md` is the compatibility contract for generated NumPy code.
