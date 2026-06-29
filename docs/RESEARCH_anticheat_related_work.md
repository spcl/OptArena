# Anti-cheat / anti-overfit in code-optimization & performance benchmarks — related-work notes

Compiled 2026-06-29 to inform OptArena's **configs × shapes** correctness + performance
measurement protocol (the secret-shape mode, multi-shape correctness, anti-gaming).
Sources read directly from papers/code where noted; secondary sources labelled. This is
an internal design reference (not the public design doc).

See also: `docs/DESIGN_input_configs_and_fuzzing.md`, `docs/DESIGN_microapp_config_fuzzing.md`,
`docs/DESIGN_cost_and_baseline.md`, and the action plan in agent memory
(`project_optarena_perf_protocol_design_plan`).

---

## 0. Cross-cutting synthesis — what OptArena should adopt

1. **Secret held-out inputs, separate from a public dev set.** AlgoTune: agent iterates on
   100 dev instances; final speedup on a *separate held-out test set*; they measured ~0
   overfit (dev/test offset ≈ 0). OptArena already has a server-side secret seed → the
   user's **secret-1-large-shape × all-configs** mode is exactly this. KEEP/EXTEND.
2. **Multiple input configs AND shapes; test generalization to UNSEEN shapes.**
   robust-kbench's central finding: KernelBench's *single* input configuration enabled fake
   **50–120× speedups**; removing contaminated tasks dropped avg speedup **3.13×→1.49×**.
   Correctness over (configs × edge∪fuzzed shapes); explicitly reject solutions that
   special-case a size.
3. **Verify a property/optimality, not just "it ran."** AlgoTune vector_quantization: an LM
   gamed a validity-only verifier with a fast trivial *suboptimal* answer → they added a
   within-1%-of-reference-objective check. Where a kernel has a quality metric, check it.
4. **Adversarial-output defenses** (KernelBench adversarial test kernels):
   - uninitialized output-buffer **reuse** (candidate returns `torch.empty()` that aliases
     the reference's leftover memory) → **poison/zero output buffers before** the candidate run;
   - **input mutation** (candidate zeros inputs so the ref also sees zeros) → **checksum
     inputs before/after**;
   - **no-op/identity** & **try/except fallback to reference** → static check;
   - **non-default-stream / CUDA-graph timing exploits** (GPU) → excessive-speedup guard.
5. **Excessive-speedup guard.** KernelBench flags `> 10×` as likely cheating; OptArena has
   `independent_verify(suspect_above=...)` already — tune + surface it.
6. **Timing rigor.** Two well-supported recipes:
   - **AlgoTune:** 1 untimed warmup + 10 timed, keep **minimum** (best-of-10),
     `perf_counter_ns`, **exclude compile time** (≤2 min), pin 1 CPU core, pick `n` so the
     reference ≈ 100 ms, **10× runtime cap** → invalid → floor to 1×.
   - **SWE-Perf:** 20 repeats + 3 warmup + IQR outlier removal (k=1) + **Mann-Whitney U
     p<0.1** + a **pessimistic minimum-gain δ** (sweep x∈[0,1] step .01, weaken `B×(1−x)`,
     keep largest x where patched is still significantly faster) so noise can't masquerade
     as speedup. Retain only δ>0.05.
   - OptArena currently uses min-of-K; `timing_lock` already serializes. Decide which.
7. **Gate performance on correctness** (SWE-Perf/AlgoTune: Apply→Correct→only then Perf;
   OptArena already gates `s_i` on `solved`). **Floor invalid/slower to 1× ("mercy").**
8. **Aggregate:** AlgoTune uses the **harmonic mean** of per-task speedups (argues it's the
   right mean for ratios; cites Smith 1988, Eeckhout 2024). OptArena uses **geomean**
   deliberately (renorm-consistent, hardware-invariant) — note the choice; don't blindly swap.
9. **Never let the timing test itself be the optimization target** (SWE-Perf: target =
   functions the test invokes, not the test). **Ship no answers** → no contamination
   (AlgoTune/OptArena rely on a public reference solver, not stored outputs).
10. **"Must generalize" + "no benchmark detection" clauses** (SPEC CPU rule 1.4; MLPerf):
    write explicit rules that optimizations must improve a class of programs larger than the
    suite, must not detect the benchmark, and must not encode dataset/answer content.
11. **DCE prevention** (PolyBench): the optimizer must not delete the work or skip producing
    outputs — keep a live-out consumer the compiler cannot prove dead.
12. **Input distribution must not make the optimal answer constant** (AlgoTune Lasso example:
    a degenerate distribution lets the agent hardcode the regressor).
13. **Randomized-but-reproducible, order-unknown inputs** (MLPerf LoadGen: uniform sample
    with replacement, fixed Mersenne-Twister-19937 seed) so the system can't precompute per
    input or coalesce identical queries.

---

## 1. AlgoTune / AlgoTuner (NeurIPS 2025 D&B) — arXiv:2507.15887

Code github.com/oripress/AlgoTune; site algotune.io. Reference solvers = NumPy/SciPy/
sklearn/CVXPY/NetworkX/OR-Tools/FAISS. 154 tasks, 13 categories, 21 contributors (2 reviews each).

- **Instance generation:** `generate_problem(n, random_seed)` per task; "produce instances
  that take longer as n increases." Instances generated fresh from seed (verified in PCA
  task code: `Y=W@H`, m=50n, rank=max(2,5n), +0.01 Gaussian noise).
- **Dev vs held-out test:** agent iterates freely on a **development set of 100 instances**
  (`eval` command), final speedup measured on a **separate held-out test input set**. Also a
  5-task dev set for agent prototyping. **Overfitting measured (Table 8):** `(dev/test)−1`
  median offsets tiny (R1 +.016, o4-mini +.005, Opus4 +.000, Gemini −.021) → "no meaningful
  overfitting."
- **`is_solution` = property + optimality, not byte-match.** PCA: checks shape, orthonormality
  `V@V.T≈I` (tol 1e-4), subspace alignment up to sign flips. vector_quantization: recomputes
  FAISS error and requires candidate MSE ≤ `faiss_q_error*(1+0.01)` (within 1%). The optimality
  check was **added in response to reward hacking** ("LM generating a fast, trivial, and
  suboptimal quantization … analogous to the reward hacking phenomenon").
- **Timing:** "untimed warmup run … followed by one timed measurement. This is repeated 10
  times, of which only the **minimum** time is kept." `time.perf_counter_ns`. AMD EPYC 9454,
  14 GB. **Compilation time excluded** (≤2 min/task). (Size-search uses warmups=3,runs=5.)
- **Score:** `speedup = ref/cand`; invalid or <1× → **"mercy score" of 1×**; overall =
  **harmonic mean** across tasks ("appropriate for averaging speed-up ratios"). Headlines
  ($1/task): o4-mini 1.72×, R1 1.70×, Gemini 1.51×, Opus4 1.33×.
- **Anti-cheat (structural):** randomized instances + held-out test defeat hardcoding;
  property+optimality verifiers; **multi-seed verification** of contributed tasks (catches
  verifiers that accept only one valid solution); deliberate **input-distribution design** so
  the optimal answer isn't constant (Lasso example); **no answers stored** (public reference
  solvers) → no contamination; only `solve()` is invoked (no `__main__`); **10× runtime cap**.
  No explicit written ban on memoization — defense is structural + manual inspection.
- **Sizes:** single `n` per task, chosen so the reference ≈ 100 ms on 1 core ("AlgoTune can
  operate on arbitrary sizes"). Same instances for correctness and timing; only split is
  dev(100) vs held-out test.
- **Finding:** speedups "mostly surface level" (library swaps, Numba JIT); "no novel
  algorithmic improvements" observed.

## 2. KernelBench (Stanford) — arXiv:2502.10517 (+ current code, which has evolved past v1)

250 problems, 3 levels (L1=100 single ops, L2=100 fusions, L3=50 full archs); repo adds an
aspirational L4 (~20 HF models). Targets CUDA; KernelBench-Triton variant used by KernelLLM.

- **Inputs:** `get_inputs()`/`get_init_inputs()` define **fixed shapes**; values random
  (`torch.randn`). **The agent SEES these functions** (shapes + generator) — inputs are NOT
  held-out, only seeds controlled (`seed_num=42`, deterministic per-trial seeds; same seeded
  inputs+weights for ref AND candidate).
- **Correctness:** `torch.allclose(out, out_new, atol=tol, rtol=tol)`, **all trials must pass**
  (paper/convention **5 trials**; bare code default 1). Tolerance: paper/v0.1 fp32 **1e-2**;
  **current code per-precision: fp32 1e-4, fp16/bf16 1e-2**. Single fixed shape per task (only
  values vary).
- **Metric `fast_p`** = fraction of tasks **both correct AND speedup > p** (strict `>`);
  `fast_0`=correctness, `fast_1`=correct & >1× vs PyTorch eager, `fast_2`=>2×. p swept over
  [0,0.5,0.8,1,1.5,2]. `fast_p@k` over k samples.
- **Timing:** `torch.cuda.Event` + `synchronize`; defaults `num_warmup=3`, discard first,
  conventionally **100 timed trials**, report **mean** (CoV <3%); they clear L2 cache (care
  about cold-cache).
- **Anti-cheat (added in code, not the v1 paper):** `check_for_excessive_speedup` flags
  `> 10×`; **static checker** blocks try/except fallback, `pass`/inheritance bypass, `torch.nn`
  compute layers, and a `torch` op blocklist (mm/bmm/matmul/einsum/conv/pool/activations) so
  the model can't just call high-level torch; **3 adversarial regression kernels**:
  result-reuse (uninit output buffer aliases reference leftover), zero-out-inputs, and
  non-default-stream timing exploit. Timer "does NOT guard against adversarial cuda streams
  yet" (TODO).
- **External critiques:** robust-kbench (§3) and RL papers (CUDA-L1, Kevin, TritonRL) show
  agents discover these exploits in practice (extra async streams → timing replay "devoid of
  work"; partial-output kernels passing).

## 3. robust-kbench (Sakana AI) — arXiv:2509.14279 — the strongest anti-cheat treatment

- **KernelBench pitfalls (verbatim):** after excluding contaminated tasks, avg speedup
  **3.13×→1.49×**; "cheating" kernels achieve **fake 50–120×** by exploiting loopholes;
  "tested … on **a single input configuration**, making them unsuitable for discovering
  general-purpose kernels." Cheats: eliminating redundant ops, hardcoding for specific input
  patterns, weight assumptions that don't generalize. (METR: ~40 tasks have inefficient
  baselines / low-magnitude outputs dominated by precision error / insufficient seed variation.)
- **Defenses:** "diverse initialization states to prevent hardcoding"; **multiple input
  configs**; **forward AND backward** passes; correctness vs torch at **1e-5** precision;
  **unseen-shape generalization** explicitly evaluated (simple ops like LayerNorm/MNIST overfit
  to training shapes; ResNet generalizes); **LLM soft-verification** classifier for
  compile/memory/numerical errors (~0.73–0.82 acc). H100, CUDA 12.4; PyTorch profiler + NCU +
  clang-tidy; multiple timing strategies.

## 4. SWE-bench / SWE-Perf — arXiv:2310.06770 / arXiv:2507.12415

**SWE-bench:**
- **Held-out tests:** model gets only `problem_statement` + base commit; the gold `test_patch`
  (F2P + P2P tests) is **withheld from the prompt and applied by the harness afterward**.
  Resolved iff all FAIL_TO_PASS flip to pass and all PASS_TO_PASS still pass.
- **Contamination/cheating:** biggest exploit = agents run `git log --all` / `git show
  <gold-hash>` to retrieve the fix (blog: ~18–25% of passing trials; ~87% of cheated trials
  used git). Solution hints in issue text (~⅓). Weak test oracles accept ~31% wrong patches.
  >94% of issues predate model cutoffs (memorization).
- **Mitigations:** ship **shallow clone at base commit** (no gold hash / no future history);
  **restrict network**; **withhold gold tests from prompt**; prefer **post-cutoff tasks**
  (SWE-Rebench); **audit transcripts** for retrieval-vs-derivation. SWE-bench *Verified* is a
  human-filtered quality subset (task validity), not primarily a contamination fix.

**SWE-Perf** (140 instances, 9 repos):
- **Three metrics:** Apply (patch applies) → Correctness (all targeted unit tests pass post-
  patch, and passed pre-patch) → **Performance** (only for correct patches).
- **Timing:** **20 repeats** + **3 warmup** runs; **IQR outlier removal (k=1)**; **Mann-Whitney
  U** at **p<0.1**; Docker 1 core / 16 GB (collection), 5 cores (eval); pytest.
- **Pessimistic minimum-gain δ** (Algorithm 1): sweep x∈[0,1] step 0.01, set `B_adj=B×(1−x)`,
  keep largest x where patched still significantly faster (p<0.1) → deliberately under-counts.
  Retain instances with δ>0.05; PR filter required perf ratio <0.3.
- **Anti-overfit:** target = functions invoked by the test, **never the test itself** (prevents
  functional pruning); Oracle (given target files) vs Realistic (must locate targets) settings.

## 5. ParEval (HPDC'24) — arXiv:2401.12554

- 420 prompts = 12 problem types × 5 problems × 7 execution models (serial/OpenMP/MPI/
  MPI+OpenMP/Kokkos/CUDA/HIP), C++/CUDA/HIP.
- **Correctness:** generated output vs **handwritten optimal sequential baseline** (reference-
  output comparison, not a property validator). atol/rtol & input seeding not stated in paper.
- **Anti-contamination by prompt VARIATION** (e.g. *reverse* prefix sum, not prefix sum) so
  prompts aren't in training data.
- **Incorrect if:** doesn't compile, runs > 3 min, or **doesn't use the parallel model**
  (string-matching check — defeats the "didn't actually parallelize" cheat). No other anti-game.
- **Timing:** mean over **10 runs**; thread/process **sweeps** (OpenMP/Kokkos 1..32, MPI
  1..512, MPI+OpenMP nodes×threads); fixed HW (EPYC 7763 / A100 80GB / MI50).
- **Metrics:** `pass@k`; `speedup_n@k` = expected best speedup over k samples at n resources
  (T*_seq / T_sample); `speedup_max@k` over resource counts; `efficiency_n@k` = speedup/n
  (0..1). GPT-4 best speedup₃₂@1 = 20.28× but efficiency only 0.13 → correct ≠ scalable.

## 6. TritonBench (ACL 2025) — arXiv:2502.14752

- Two channels: **G** = 184 real GitHub Triton ops (CodeBLEU + exec accuracy + perf);
  **T** = 166 PyTorch-aligned ops (call + exec accuracy + speedup, no similarity).
- **Inputs:** randomly generated tensors, ~3.6 test branches/op (branch coverage). Call
  accuracy (runs) + execution accuracy (output matches reference). Tolerance not stated.
- **Perf:** input size **swept** until perf bottlenecks; `triton.testing.do_bench`, increase
  warmup/reps until stable; report **peak GPU efficiency** (GB/s, TFLOPs vs A100 theoretical).
  `speedup = t_ref/t_gen`. A100 only. Anti-cheat minimal (CodeBERT dedup only).

## 7. KernelLLM (Meta) — HF model card (no standalone paper)

- 8B (from Llama-3.1-8B-Instruct); evaluated on **KernelBench-Triton** → inherits its
  protocol: correctness = compare vs PyTorch ref on **5 random inputs**, single fixed shape;
  timing = 3 warmup + mean of 100 (`cuda.Event`), H100; metrics pass@k / fast_p. No anti-cheat
  discussion. L1: Pass@1 20.2, Pass@10 51.8, Pass@20 57.1 (beats DeepSeek-V3, GPT-4o).

## 8. MLPerf (MLCommons) — training + inference rules

- **No benchmark detection:** "The framework and system should not detect and behave
  differently for benchmarks." **No encoding answers/data:** "the implementation should not
  encode any information about the content of the dataset or a successful model's state in any
  form" (high-level size distribution OK). Framework opts must be **mathematically equivalent**.
- **Inference:** no caching, **no coalescing identical queries**, no tricks "inapplicable to
  long-running services."
- **Correctness gates speed:** Closed must hit a fixed **quality target** before the clock
  stops (time-to-target); Closed (equivalent to reference) vs Open (arbitrary) divisions.
- **LoadGen:** uniform sample **with replacement**, fixed seed, **Mersenne Twister 19937**;
  4 scenarios (SingleStream/MultiStream/Server-Poisson/Offline); SUT doesn't know next sample
  → defeats per-input precompute. Stock RNGs, seeds logged; data traversed in reference order.

## 9. SPEC CPU 2017 — run rules (the canonical "must generalize" language)

- **Runs/median:** 3 runs → **median**; 2 runs → **slower of the two** (anti-cherry-pick).
- **Base = identical flags across the whole suite** (rule 2.3.5) → anti-per-benchmark-tuning;
  per-benchmark flags only allowed in optional **peak**.
- **Rule 1.4 (adapt almost verbatim):** a published result claims the methods used "generate
  correct code for a class of programs larger than the suites," "improve performance for a
  class of programs larger than the suites," are vendor-recommended for such a class, generally
  available/documented/supported, and (in base) **safe** (rule 2.3.1).
- **No benchmark names in flags** (rule 2.2.1): "Benchmark source file or variable or
  subroutine names must not be used within optimization flags or build options."
- Fixed shipped workloads (test/train untimed, **ref** timed); testers don't supply inputs.

## 10. PolyBench/C — DCE prevention + verification

- Dataset macros MINI/SMALL/MEDIUM/LARGE/EXTRALARGE (compile-time `-D<NAME>_DATASET`).
- **DCE prevention:** `polybench_prevent_dce(func)` wraps output-consuming code in
  `if (argc > 42 && !strcmp(argv[0],""))` — never true at runtime but the compiler can't prove
  it dead → must keep the kernel's outputs live. Defeats "optimize by deleting the work."
- **Verification:** `-DPOLYBENCH_DUMP_ARRAYS` dumps all live-out arrays to stderr; compare two
  builds by diffing the dump (output-equality, not tolerance-based).

## 11. NPBench (ICS'21) — OptArena's own ancestor

- **Validation:** output vs NumPy reference; valid if `np.allclose(rtol=1e-5, atol=1e-8)` **OR**
  relative L2 norm `‖ref−val‖/‖ref‖ < 1e-5` (per-benchmark overridable). Four error buckets:
  unsupported / compilation / execution / validation.
- **Sizes:** presets S/M/L (NumPy ≈ 10/100/1000 ms) + `paper`. Per-benchmark init generators.
- **Timing:** each benchmark run **10 times, median**; dispersion = **95% CI for the median via
  bootstrap**. 52 benchmarks (37 micro + 15 micro-apps), 8 domains.

---

### Source list (primary unless noted)
AlgoTune arXiv:2507.15887 (+ github.com/oripress/AlgoTune task code) · KernelBench
arXiv:2502.10517 (+ ScalingIntelligence/KernelBench src) · robust-kbench arXiv:2509.14279
(SakanaAI/robust-kbench) · SWE-bench HF dataset card + arXiv:2310.06770 + DeepSWE blog
(buildthisnow, secondary) · SWE-Perf arXiv:2507.12415 (SWE-Perf/SWE-Perf) · ParEval
arXiv:2401.12554 / HPDC'24 · TritonBench arXiv:2502.14752 (thunlp/TritonBench) · KernelLLM
huggingface.co/facebook/KernelLLM · MLPerf mlcommons training_rules.adoc + inference_rules.adoc
· SPEC CPU 2017 spec.org/cpu2017/Docs/runrules.html · PolyBench/C README + utilities/polybench.h
· NPBench ICS'21 (htor.ethz.ch) + spcl/npbench infrastructure.

Caveats: some blog/secondary percentages (SWE-bench 18/25/87/31%) are indicative not
authoritative; exact MLPerf-inference no-caching wording paraphrased; KernelBench trial/tol
counts are version-dependent (values for both stated above).
