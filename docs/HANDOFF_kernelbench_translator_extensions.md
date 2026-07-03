# Hand-off: NumPy-translator extensions for the KernelBench ML kernels (PR #5)

## Context

PR #5 (`PytorchToNP`, branch `pr5`) lands the KernelBench level-1/2/3 models plus
five converted benchmark kernels under `optarena/benchmarks/ml/`
(`conv2d_bias`, `lenet`, `mlp`, `resnet`, `softmax`). The level-1/2/3 files are the
raw PyTorch sources; **we only support their NumPy equivalents** — the PyTorch
nodes (`nn.Linear`, `nn.ReLU`, `.view`, `nn.Parameter`, `nn.Sequential`) are
converted away and are NOT the translator's concern. This doc enumerates the NumPy
nodes the converted kernels need, and splits the work in two.

Do **not** merge PR #5. This is translator-side enablement only.

## Already supported (do not redo)

Confirmed present in `numpyto_common` / `numpyto_c` / `numpyto_fortran`:

- Axis reductions with `axis=` **and `keepdims`** (`frontend.py:_read_axis_keepdims`)
  — `mean`/`var`/`std`/`sum`/`max`/`min`. This is the softmax / batchnorm /
  groupnorm / layernorm backbone, so those normalize+activate kernels are mostly
  covered already.
- **Batched** `np.matmul` / `@` over leading axes (`lib_nodes.py:359` broadcast of
  `(...,M,K)@(...,K,N)`) — attention / batched-linear.
- `np.newaxis` broadcasting, `reshape`, `transpose`/`.T`, `where`, `clip`, `maximum`,
  `minimum`, `exp`, `log`, `sqrt`, `tanh`, `abs`, `pad`, `einsum`, `concatenate`
  (basic), `cumsum`, `cumprod`, `triu`, `tril`, `arange`, `argmax`, `argmin`,
  `full`, `zeros`/`ones`/`empty`(`_like`), `power`, `flip`.

So the 5 converted kernels lean almost entirely on already-supported ops; the gaps
below are what the broader 253-model corpus will hit as it converts.

## Gaps — the nodes to add

Each item: what · why (kernels) · where · approach · effort.

### Shape-manipulation aliases (all MISSING — no mention in the translator)

1. **`np.expand_dims(a, axis)`** · norm/broadcast reshapes · desugar → `reshape`
   inserting a size-1 dim at `axis`. EASY.
2. **`np.squeeze(a[, axis])`** · pooling / head reshapes · desugar → `reshape`
   dropping size-1 dims (all, or the named axis). EASY.
3. **`np.swapaxes(a, i, j)`** · attention (Q/K transpose), channel moves · desugar →
   `transpose` with a permutation that swaps `i`,`j`. EASY.
4. **`np.take(a, idx[, axis])`** · embeddings / gather · desugar → the existing
   fancy-index/gather lowering along `axis`. MEDIUM (axis handling).

Home: the `("np", <fn>)` dispatch in `numpyto_common/lib_nodes.py` (see the
`cumsum`/`tril` entries around line 4465) and/or `numpy_desugar.py`. 1–3 are pure
shape rewrites to already-supported nodes; 4 reuses gather.

### `None` = "is this array allocated" (the optional-array pattern)

Appears in the QE `vexx_k` microapp and **63×** across the incoming corpus
(`is None` 35, `is not None` 16, `else None` 12). The translator raises
`NotImplementedError: literal None` today. Treat `None` on an array-typed value as
**unallocated** (NULL / not-associated):

5. **Unprovided optional arg** (signature default `None`, not in `input_args`) is
   `None` at emit time → fold `arg is None` → `True`, `arg is not None` → `False`,
   then dead-branch-eliminate. Kills `at_ = np.eye(3) if at is None else ...`,
   `if coulomb_fac_q is not None:`, etc. MEDIUM.
6. **Conditional-None allocation** `X = alloc(...) if cond else None` →
   `X = alloc(...)` (always allocate). Sound because a valid kernel only *reads* `X`
   under the same `cond` guard (reading it when `not cond` would be a `None`-index
   crash), so the extra buffer is never observed. Removes the `np.zeros(...)`-in-
   ternary that blocks `vexx_k`. MEDIUM.
7. **`None` as a value / call arg** that survives folding → emit as `NULL`
   (C) / unallocated (Fortran), with `is None`/`is not None` as the null test.
   MEDIUM; couples with 5–6.

Home: a small dedicated desugar pass (run before C/Fortran emit), plus a
`Constant(None)`→NULL case in `numpyto_c/emit.py:emit_expr` and the Fortran twin.
Ground truth to fix against: `optarena/benchmarks/hpc/spectral_methods/vexx/vexx_k_numpy.py`
(lines 447/448/480/491/515 and the `vcut_corrected` arg passed at ~503).

### Array-combining

8. **`np.stack`** · residual/channel concat, attention head merge · currently
   PARTIAL: `expand_hstack`/`_parse_stack_concat` (`lib_nodes.py:2160`,`2250`) only
   accept `Name` operands with statically-known shapes and otherwise raise. Generalize
   to N operands + a new leading axis. MEDIUM.
9. **`np.concatenate`** · same · basic path exists (`lib_nodes.py:499`); **verify +
   harden** for the incoming cases (N arrays, non-0 axis, computed shapes). MEDIUM.

### ML numerics + end-to-end

10. **GELU / erf** · transformer MLP blocks · confirm `erf` is available to
    C/Fortran (else add via `std::erf`/`erf`/`erfc` or the tanh approximation the
    model uses). LOW–MEDIUM.
11. **End-to-end emit+validate** the 5 converted kernels (`conv2d_bias`, `lenet`,
    `mlp`, `resnet`, `softmax`) on C / C++ / Fortran through the numerical oracle;
    fix whatever surfaces. This is the acceptance test for the whole effort. MEDIUM,
    ongoing.

## The split (independent halves)

**Half A — Shape & `None` desugars** *(owner: this chat)*
- Items **1–4** (expand_dims, squeeze, swapaxes, take)
- Items **5–7** (the `None`/optional-array allocation-check desugar; un-blocks `vexx_k`)

**Half B — Array-combine & ML numerics + validation** *(owner: other chat)*
- Items **8–9** (stack generalize, concatenate harden)
- Item **10** (GELU / erf)
- Item **11** (end-to-end emit+validate the 5 ml kernels; own the acceptance gate)

Rationale: Half A is one cohesive desugar surface (shape rewrites + a None pass),
touching mostly a new pass + the `("np",...)` dispatch. Half B owns the combining
nodes and the end-to-end validation loop. The one shared file is
`lib_nodes.py` (A adds shape entries near 4465; B edits stack/concatenate near
2160–2260) — non-overlapping regions, but coordinate to avoid churn.

## Constraints

- NumPy equivalents only; never add PyTorch-node support.
- Every new node gets a unit test (desugar in isolation) + a faithfulness test, per
  the repo rule "each port = init.shapes + faithfulness test".
- Don't hardcode dtypes; infer from the registry.
- Don't merge PR #5.
