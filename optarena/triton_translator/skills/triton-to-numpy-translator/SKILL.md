---
name: triton-to-numpy-translator
description: Translate Triton benchmark Python files into standalone NumpyToC-compatible numpy, build or improve the translator under src, generate separate Triton result outputs, and create tests for generated numpy behavior.
---

# Triton To NumPy Translator

## Operating Contract

Use this skill when working on this repository's Triton-to-numpy translator. Treat `CONTRIBUTOR_GUIDE.md` as the compatibility contract for generated numpy: outputs should be static-shape and buffer-oriented where practical, free of `torch`, `triton`, and project-local imports, and limited to the numpy/control-flow surface documented there.

Generated Triton results are meant to be consumed later by NumpyToC/NumpyToFortran as isolated files. Each generated file must therefore be a clean, minimal, standalone numpy implementation of the original kernel math. Do not emit shared runtime imports, helper-library imports, broad generic compatibility layers, or unused helper code into result files. Inline only the math needed by that specific source file.

Do not weaken the contributor guide to make a conversion pass. If an original Triton feature cannot be represented within the guide, stop and explain the missing rule before editing the guide.

## Required Layout

- Build translator code under `src/`.
- Write converted Triton outputs under a separate result tree, currently `triton_result/TritonBench_G_v1/`.
- Put Triton-specific test infrastructure under `test/`.
- Keep source inputs in `TritonBench_G_v1/` unchanged unless the user explicitly asks otherwise.
- Keep generated kernels numpy-only. Tests may import `torch` or `triton` only if needed, but generated result files may not.

## Translation Workflow

1. Read representative Triton kernels before changing translator logic.
2. Prefer wrapper-level semantics when the Python wrapper clearly describes the output shape and operation.
3. For simple `@triton.jit` kernels, translate `tl.load`/`tl.store` math into direct numpy operations or explicit loops.
4. Generate a result for every source file. If a wrapper is unsupported, emit a minimal function that raises `NotImplementedError` with the missing construct.
5. Track translation status in a sidecar status JSON so unsupported cases are visible and iterable.
6. Run syntax checks and generated-output tests after each translator change.

## Conversion Rules

- Replace `torch` allocations with numpy allocations only when the original wrapper's semantics are clear.
- Replace `tl.arange`, masks, `tl.load`, and `tl.store` with direct numpy slicing, broadcasting, or explicit loops.
- Replace `tl.dot` with `np.matmul` or `@` for supported matrix forms.
- Replace `tl.where`, reductions, and elementwise math with the corresponding numpy operations from `CONTRIBUTOR_GUIDE.md`.
- Strip Triton launch configuration, grids, `tl.constexpr`, `num_warps`, and block-size tuning from generated files unless the value affects math.
- Preserve public wrapper function names where possible; generated files should be usable by calling the same wrapper name with numpy arrays.
- Do not emit code that depends on CUDA, Triton runtime behavior, or external helper files.

## Test Expectations

- Tests should import generated Triton result files dynamically and exercise translated functions on CPU numpy arrays.
- Where the operation is recognized, compare against an independent numpy reference.
- Where the operation is unsupported, assert that the generated function raises `NotImplementedError` rather than silently returning wrong values.
- Keep test inputs CPU-sized and representative, not trivially small.
- Report per-file failures with enough context to improve the translator quickly: syntax error, unsupported function, shape mismatch, or numerical mismatch.

## Project References

- Read `CONTRIBUTOR_GUIDE.md` for the allowed generated-numpy surface.
- Read representative files in `TritonBench_G_v1/` before implementing new translation patterns.
