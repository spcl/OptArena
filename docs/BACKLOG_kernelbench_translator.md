# Backlog: numpy-translator work for the KernelBench PR

Scope of what the KernelBench PR (`origin/PytorchToNP`, PR #5) will need from the
numpy->backend translators. The PR lands 250 raw-PyTorch models under
`optarena/PytorchToNumpy/level{1,2,3}/` (level1 = 100 single-op, level2 = 100
fused, level3 = 50 architectures); only 5 are converted to numpy + registered
today (`optarena/benchmarks/ml/`: conv2d_bias, lenet, mlp, resnet, softmax).
Only their NumPy equivalents are the translator's concern -- the `nn.*` / `F.*`
PyTorch nodes are converted away.

Method: two static-inventory passes over the level1 and level2/3 corpora plus an
empirical probe that ran ~40 ML numpy patterns through every backend
(c/cpp/fortran/numba/pythran/jax) via the op-oracle, then a manual
reconciliation of every flagged failure (several were probe-harness artifacts,
not real gaps -- see below). Dates are 2026-07-06.

## Already supported -- do NOT re-implement

Every one of these emits + validates bit-exact on the native backends
(c/cpp/fortran) and jax today (verified through the op-oracle at tiny sizes):

- **Convolution**: conv2d stride-1 (windowed `x[:,i:i+K,j:j+K,:,None]*w[None]`
  reduced with `sum(axis=(1,2,3))`); ConvTranspose via `np.add.at` scatter.
- **Pooling**: maxpool / avgpool (tuple-axis `max`/`mean` over windows), adaptive
  avg pool (integer-division window bounds), global avg pool (keepdims).
- **Normalization backbone**: BatchNorm inference, `mean`/`std` with `axis=` +
  `keepdims`, the affine `(x-m)/sqrt(v+eps)*g+b`. (var-with-axis is the one hole
  -- see gaps.)
- **Attention primitives**: batched matmul over leading axes -- both 3-D and
  **4-D** `@` validate bit-exact (the probe's "4-D matmul miscompiles" was an
  artifact of its own MHA reference); `np.swapaxes` (Q/K transpose) lowers fine
  (the probe's "swapaxes unimplemented" was likewise its kernel, not the
  translator); softmax with `axis=-1, keepdims=True`; `np.einsum` incl. ellipsis.
- **Activations**: gelu (erf and tanh forms; `erf` is available), sigmoid, tanh,
  silu, leaky_relu, elu, hardswish, mish, hardtanh, softplus -- all elementwise.
- **Recurrence**: LSTM cell + a multi-step time loop carrying `(c, h)`, GRU cell.
- **Shape / combine**: `np.concatenate` (axis 0 and 1), `reshape`, `tril`/`triu`,
  `np.roll`, `cumsum(axis=)`, `F.normalize` (axis L2), matrix `norm(A,1/inf)`,
  axis-L2 `norm(x,axis=)`.

Implication: the hard-looking composite ops (ConvTranspose index math, grouped
conv, the masked-softmax attention chain, RNN state-carry loops, group-reshape
norms, Mamba einsum/cumsum) are **conversion + registration work, not translator
features** -- their numpy lowerings already compile.

## Genuine translator gaps (native c/cpp/fortran), ranked

1. **`np.var(x, axis=k, keepdims=...)` fails to compile** -- emits an undeclared
   reduction buffer (`__cb1`/`__cb2`/`__cb3` undeclared in C/C++; "function
   result on lhs must have the pointer attribute" in Fortran). Isolated: `np.std`
   with an axis compiles, scalar `np.var(x)` compiles, and a LayerNorm rewritten
   with manual variance (`np.mean((x-m)**2, axis, keepdims)`) validates bit-exact
   -- so it is specifically `np.var`'s **axis** lowering (the post-fn temp for the
   two-pass mean-then-sumsq) that drops a declaration. **This is the single root
   cause blocking LayerNorm / GroupNorm / InstanceNorm (~40 level2+level3
   models).** Likely same `__cb` post-reduction-temp declaration path that was
   fixed for axis-norm; mirror that fix for the var expander. HIGH.
2. **`np.take_along_axis`** -- `NotImplementedError` at emit on all native
   backends. Needed by CrossEntropy-style gather / max-value select (~4 models).
   MEDIUM.
3. **`np.pad(x, ..., constant_values=-np.inf)`** -- emits but pads with 0, not
   `-inf` (wrong result), so a maxpool-with-explicit-pad reads 0 at the border.
   Most maxpool numpy ports avoid the pad (window-max directly), so this is
   LOW; fix = honor a non-zero/`-inf` `constant_values` in the pad expander.
4. **Indexing a compound expression, e.g. `(m + np.log(...))[:, 0]`** --
   `NotImplementedError`; a stable LogSumExp written inline hits it. Workaround is
   trivial (bind the expression to a temp first), so LOW; a general fix is to
   hoist a subscript's non-Name base to a temp before lowering.

## Python-JIT backend status

- **numba: FIXED this session for the ML reductions.** The reduce-axis desugar
  now lowers `sum`/`prod`/`mean`/`var`/`min`/`max`/`argmin`/`argmax` with a
  negative axis (`axis=-1`), `keepdims=True`, and a **tuple axis** (`axis=(1,2,3)`)
  to explicit loops numba can njit; the `@nb.njit` decorator now lands on the
  kernel + all helper defs (not just the first def); list-literal shapes
  (`np.empty([...])`) are rewritten to tuples; and helper param ranks are inferred
  from body usage + return-rank propagation so multi-helper kernels (lenet/mlp/
  resnet) resolve. All 5 registered ml kernels are numba-`ok`. Residual numba
  skips only track the native gaps above (var-axis, take_along_axis).
- **pythran: FIXED this session -- all 5 ml kernels now `ok`** (was 3/5). The two
  holdouts were pythran lazy-`numpy_expr` limits, not core gaps:
  - lenet -- `np.reshape` of a computed array ("Unsupported attribute 'reshape'").
  - mlp -- indexing a broadcast-add expression (`x @ w + b`) passed through a
    helper call; pythran cannot `_index` a lazy `broadcasted<>` numpy_expr.
  Fixed by `_PythranMaterialize` in `numpyto_pythran/emit.py` (`_clean_for_pythran`):
  force evaluation with `np.ascontiguousarray` -- `np.reshape(X, s)` ->
  `np.ascontiguousarray(X).reshape(s)`, and a compound arg to a LOCAL helper call
  is wrapped `helper(np.ascontiguousarray(expr))`. (`.copy()` does NOT work --
  pythran cannot copy a `numpy_expr`.) Test: `test_pythran_materialize.py`.
- **op-oracle pythran harness bug (not a translator gap):** `_op_oracle`'s
  synthesized bench_info lets `parse_kernel` promote the size symbols into
  `kir.input_args`, so `emit_pythran` writes `#pythran export f(int, int, ...)`
  with more args than the verbatim `def f(x, out)` -> "Too many arguments when
  exporting f" -> every op-oracle pythran probe silently `skip`s. Masked until now
  by skip-tolerant assertions. Fix = give the op-oracle's verbatim-signature
  arg list without the shape symbols (they are ABI params, not def params).

## Non-translator work (the bulk of the PR)

- **Convert + register the remaining 245 models.** Each = a faithful
  `<short>_numpy.py` + a `<short>.yaml` manifest + `init.shapes` + a faithfulness
  test. The numpy lowerings for the dominant op families (conv 35, ConvTranspose
  52, pooling 50+, norms 40+, attention 10, RNN 12, activations, concat 20)
  already compile, so this is porting, not translator work.
- Watch items when porting: running-stat BatchNorm (read `running_mean/var`, do
  not recompute batch stats); group-reshape for GroupNorm/InstanceNorm; the
  Mamba2 models (5-index einsum + cumsum + tril-masked segsum) are the most
  exotic; Swin needs 6-D reshape/transpose + `np.roll` + pad window partition.

## Fortran batched-einsum size-symbol non-determinism (watch)

`docs/BACKLOG.md` records a flaky Fortran emit that types a size symbol REAL
instead of INTEGER (`out(N,M,B)` -> "Expression must be of INTEGER type") ~40% of
runs, hash-order dependent. Not reproduced in this session (explicit `Bij,Bjk->Bik`
and 3-D/4-D `@` passed repeatedly), but attention/transformer models lean on
batched einsum, so the defensive fix (type size symbols int64 unconditionally /
sort the symbol set in the emit's integer classification) is worth applying.
