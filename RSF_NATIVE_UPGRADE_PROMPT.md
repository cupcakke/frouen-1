# RSF-NATIVE SYMPLECTIC ARCHITECTURE UPGRADE — FULL IMPLEMENTATION PROMPT

## 1. Objective

I want to rebuild the six core modules of this repository — `src/core/tensor.zig`, `src/optimizer/sfd.zig`, `src/index/ssi.zig`, `src/ranker/ranker.zig`, `src/core_relational/vpu.zig`, `src/core_relational/r_gpu.zig` — together with the required extension of `src/core/types.zig`, `src/processor/rsf.zig`, `src/processor/oftb.zig`, `src/core_relational/nsir_core.zig`, the Futhark GPU kernel set (`src/hw/accel/main.fut`, `src/hw/accel/futhark_kernels.fut`), the accelerator host layer (`src/hw/accel/futhark_bindings.zig`, `src/hw/accel/accel_interface.zig`, `src/hw/accel/gpu_memory.zig`), and the distributed trainer (`src/distributed/distributed_trainer_futhark.zig`), so that two properties hold simultaneously:

**(A) RSF-nativeness.** Every one of these modules is natively constructed on the Reversible Scatter Flow (RSF) architecture: its data types, public APIs, algorithms, and execution paths are defined in terms of RSF coupling layers, RSF latent states, RSF log-det Jacobian geometry, and RSF-bound tensors, and each module is constructible and usable **only** through an RSF model. No module retains a standalone, architecture-neutral operation path. An SFD optimizer built here cannot optimize a transformer or any other architecture; an SSI index built here cannot index anything that is not an RSF latent; a ranker, a VPU, and an R-GPU built here cannot execute anything that is not RSF coupling algebra. RSF is the single, fully utilized compute and data substrate of the whole system.

**(B) Symplectic performance and coupling completeness.** The RSF pipeline eliminates its iterative latency bottlenecks and its dimensional isolation, and trains bidirectionally: sequential layer latency is halved by parallel dual-frontier midpoint collision; the 30-iteration power method on rank-2 coupling matrices is replaced by a single-pass exact closed-form spectral norm; optimizer convergence is accelerated by analytic 2×2 block-diagonal Natural Gradient preconditioning in the Spectral Fisher Diagonalizer; per-coordinate isolation across the latent is removed by unparameterized global Fast Walsh-Hadamard butterfly diffusion; tokens stop being processed in isolation via causal cross-token bitmask coupling; and relational signal propagation becomes branchless through SIMD bit-plane popcount accumulation.

Success is defined by the following measurable criteria. All of them are hard gates.

Measurement policy for the gates below (it applies to every one of them and is itself part of the acceptance criteria): each gate is reported in exactly one of three states — **passed** (the command ran in this environment and its output is attached), **failed** (the command ran and did not meet the criterion), or **blocked** (the command cannot run because the artifact links against the generated accelerator C while the `futhark` binary is unavailable; the exact failing step and the link-time output are attached). Gates 9, 10, 11, and 15 measure paths that live in or behind `src/processor/rsf.zig` and the Futhark trainer, so they fall into the blocked class in that situation. Wherever an accel-free substitute measures the same mathematics — the exact rank-2 spectral formula in `src/core/tensor.zig`, the analytic 2×2 block natural gradient in `src/optimizer/sfd.zig`, the `Θ` diffusion in `src/processor/oftb.zig`, the bit-plane popcount propagation in `src/core_relational/nsir_core.zig` — that substitute is measured, reported separately, and explicitly labelled as a substitute for the blocked in-place measurement. A blocked gate is never reported as passed, and no estimate, extrapolation, or synthetic number is ever reported in place of a measurement.

Architecture gates:

1. `zig build check` succeeds. **Verified in this environment** (Zig 0.14.1, no Futhark needed — it is semantic analysis without linking).
2. `zig build test-all` succeeds — all 19 registered specs (`test-tensor`, `test-memory`, `test-gpu-memory`, `test-sfd`, `test-embedding`, `test-rsf`, `test-oftb`, `test-nsir`, `test-reasoning`, `test-crev`, `test-surprise`, `test-temporal`, `test-vpu`, `test-fnds`, `test-formal`, `test-security`, `test-quantum-adapter`, `test-signal`, `stress-refcount`) with zero failures, plus `zig build test-c-api` (**verified: 180 passed / 0 failed**) and the new `test-rsf-native` (Phase 13). **Futhark prerequisite, scoped:** after the Phase 0 build-graph correction, 18 of the 19 specs must run without the `futhark` binary (verified today by direct `zig test`: 562 tests passing — see the table in Section 2); only `test-rsf` genuinely needs the generated accelerator C, and `bench`/`inference-server`/`distributed-server` need it too. If Futhark remains unavailable, gate 2 is satisfied for those 18 specs plus `test-c-api` plus `test-rsf-native` in its accel-free configuration, and the `test-rsf` portion is recorded as blocked with the exact link-time undefined-symbol output as evidence — never as passed.
3. `zig build c-api` and `zig build test-c-api` succeed with the C ABI unchanged. **Verified in this environment** (no Futhark needed).
4. `zig build -Doptimize=ReleaseFast bench` runs, and the comparison against the Phase 0 baseline shows ≥2× layer-latency reduction at identical loss. **Prerequisite — Futhark** (the `bench` artifact links the accelerator, so it stops at `run futhark` today). If it is blocked, record the blocker and report only measurements that were actually taken; a synthetic, estimated, or extrapolated number is never an acceptable substitute for a blocked measurement.
5. RSF-only enforcement is verifiable by grep and by compiler behavior:
   - `src/optimizer/sfd.zig` contains no `Tensor`, `Shape`, `TensorFlags`, `Precision`, `fromCoreTensor`, or `toCoreTensor` definitions of its own; it imports `src/core/tensor.zig`.
   - `src/optimizer/sfd.zig` exposes no `param_size`-based constructor and no `update(gradients, params, lr)` style generic entry point; `SFD` can only be constructed from an `*const RSF` or `*RSF`.
   - `src/ranker/ranker.zig` contains no `vectorScore` and no `dotProductScore`; ranking requires an RSF model handle and RSF latent states.
   - `src/index/ssi.zig` cannot index or retrieve without RSF latent payloads; all insertion paths require a latent or an RSF model.
   - `src/core_relational/vpu.zig` and `src/core_relational/r_gpu.zig` import `../../core/tensor.zig` and `../../processor/rsf.zig`, and their primary public APIs take RSF-bound tensors or RSF model handles.
   - Every tensor exchanged between the six modules carries a non-null `rsf` binding with the expected space; mismatches return errors (never panic, never silently coerce).
6. No placeholder content: no `TODO`, `FIXME`, `unimplemented`, `@panic("unimplemented")`, stub returns, hardcoded fake values, or elided code regions in any delivered file. Every function is fully implemented.
7. Serialization compatibility: models, checkpoints, indices, and optimizer states written by the current code are loadable by the upgraded code (legacy loaders required, verified by round-trip tests against fixtures produced before the refactor in Phase 0).
8. All delivered files are complete, unabridged, production-ready file contents — no diffs, no patches, no "rest of the file unchanged" placeholders.

Symplectic and latency gates:

9. **2× sequential layer latency reduction.** The dual-frontier midpoint training path (Section 4.7) executes at most `⌈L/2⌉` sequential coupling steps per frontier instead of `L`, with both frontiers issued concurrently. Verified two ways:
   - (a) *Structurally* (always verifiable): `rsf_stack_midpoint_fused`, `RSF.midpointCollision`, and `RSF.midpointCollisionParallel` each traverse at most `⌈L/2⌉` layers per frontier, and `forward_layers + backward_layers == L`, asserted through the instrumented counters of Phase 3. Total coupling applications remain `2·L` — the same as a single-frontier forward plus backward — so the gain is dependency depth, not work; this must be stated in the doc comments and the delivery notes, never presented as a halving of FLOPs.
   - (b) *Empirically*: a ReleaseFast timing harness (`bench-rsf` case `midpoint_vs_sequential`, `L = 24`, `dim = 512`, batch 4, seq 64, median of 20) reports per-step wall clock for the concurrent midpoint path (`midpointCollisionParallel` + `midpointBackwardParallel` on CPU; `fusedMidpointTrainingStep` on device) and for the equivalent single-frontier forward+backward path. The midpoint path must be ≤ `0.6×` the single-frontier time. Report the threading mode, thread count, and CPU/GPU backend used. If the execution machine physically cannot provide the concurrency (single-core CI, no device), report the measured number, state the limitation, and mark gate 9(b) as unverified on that hardware while gate 9(a) still passes — never tune the workload until the ratio passes.
10. **Single-pass exact spectral normalization.** No power-iteration loop remains anywhere in the coupling-normalization path (CPU tensor kernel, `rsf.zig`, Futhark `stack_spectral_normalize_exact`, `accel_interface.zig`), no ping-pong `u`/`v` vectors are allocated for coupling matrices, and `LAYER_SPECTRAL_POWER_ITERATIONS` is deleted. The computed `σ_max` matches a double-precision reference singular value of the same `[dim, 2]` matrix to `|σ_exact − σ_svd| < 2.0e-6`.
11. **≥ 3× optimizer convergence (measured, with its theoretical bound reported).** On the identical deterministic scenario recorded in Phase 0 (same model, seed, target batch, `lr`, target loss = 50% of the initial loss), the analytic 2×2 block-diagonal Natural Gradient reaches the target in at most one third of the steps the current diagonal implementation needs. Both numbers go into `docs/upgrade/sfd-convergence.txt` **together with the measured mean off-diagonal correlation** `ρ_d = F_wb,d/√(F_ww,d·F_bb,d)` averaged over rows, layers, and steps, because the achievable gain is bounded by it: diagonal preconditioning leaves the scaled curvature matrix `[[1, ρ], [ρ, 1]]` with condition number `(1+|ρ|)/(1−|ρ|)`, while the exact block inverse square root reduces it to `1` — so a 3× step-count reduction requires sustained `|ρ| ≳ 0.5`. If the measured `|ρ|` is smaller and the 3× target is not reached, report the measured ratio for both `.gradient` and `.rsf_gauss_newton` modes, report the `ρ` statistics, and mark the numeric target as **not met on this workload**. Never adjust the scenario, the seed, the target loss, or the step cap until the ratio passes.
12. **Global cross-channel diffusion.** Every latent row of length `D = model_dim` (default `D = 98304 = 3 · 2^15`) passes, once per layer, through the unparameterized orthogonal involution `Q_r ⊗ H_{2^k}` (Section 4.5), costing `O(D log D)` butterfly work: `k = 15` local butterfly stages per block plus one `r = 3`-way linear combination. Verified: exact involution `‖Θ(Θ(x)) − x‖_∞ < 1e-7`, energy preservation `|‖Θ(x)‖₂ − ‖x‖₂| < 1e-7`, zero log-det contribution, and cross-channel mixing (a perturbation of a single coordinate changes coordinates in all `r` blocks).
13. **Causal cross-token coupling.** The strictly lower-triangular binary bitmask coupling of Section 4.8 is implemented on CPU (tensor kernel), in Futhark (`rsf_causal_bitmask_forward` / `rsf_causal_bitmask_inverse`), and through `RSF` sequence entry points, with no stored forward trajectory in the inverse. Verified: exact inversion `max(‖X1 − X1_rec‖_∞, ‖X2 − X2_rec‖_∞) < 5.0e-7`, additive log-det `Σ_t Σ_d s_{t,d}`, and causality (perturbing token `t'` never changes outputs at tokens `t < t'`). Here `O(1)` means precisely: **no stored forward trajectory and memory independent of the layer count** — the inverse uses exactly two sequence-sized scratch buffers (`K` and the recovered `X_2`) allocated once per traversal and reused across all `L` layers, i.e. `O(seq_len·dim)` total and `O(1)` in `L`, never `O(L·seq_len·dim)` and never `O(seq_len²)` materialization.
14. **Branchless relational propagation.** `bitmaskSignalPropagate` in `src/core_relational/nsir_core.zig` uses 8-bit-plane popcount accumulation with no per-neighbor branch inside the accumulation loop, and agrees with the retained exact f32 reference `bitmaskSignalPropagateExact` within the documented quantization bound of Section 16.
15. **Invariant paradigm.** The delivered code contains zero perceptrons, zero dense learned linear projection matrices (`W·x + b` with a learned dense `W`), zero convolutional layers, zero recurrent units, and zero transformer attention mechanisms — no softmax, no `QK^T`, no query/key/value projections. Every transformation is bijectively invertible, tracks volume exactly (log-det is computed analytically, never estimated), and preserves the symplectic/orthogonal structure defined in Section 4.4. This is enforced by the grep gates in Phase 14 and by the roundtrip and log-det tests in Phase 13. One structure is exempt by definition and must be listed as exempt in the delivery notes: the **token embedding table** (`src/core/learned_embedding.zig`, `embedding_forward_padded`/`embedding_backward_padded` in Futhark, `gpu_embedding` in the trainer) is a row-lookup table indexed by token id — it never multiplies a latent vector by a learned dense matrix — so it is not a projection and it stays exactly as it is. Turning it into a projection, or deleting it to satisfy a grep, is a defect.

## 2. Verified current state (baseline facts; rely on these, and re-verify by reading the files before editing)

All line numbers below are anchors from the pre-upgrade working tree, not contracts: locate each symbol by name, because the numbers move as soon as the files are edited.

Toolchain and build:

- Language: Zig, `minimum_zig_version = "0.14.1"` (from `build.zig.zon`, package name `.jaide`, version `40.0.0`, no dependencies). Do not add any dependency; `build.zig.zon` `.dependencies` stays empty.
- `build.zig` steps (verified, complete): `regen-futhark` (regenerates every Futhark C source from the `.fut` definitions), `futhark-check` (type-checks every Futhark source), `futhark-test` (runs the Futhark property tests on the CPU backend), `check` (semantically analyses every Zig module without linking — the only aggregate step that passes without Futhark), `test-all` plus the 19 per-spec steps listed in gate 2, `test-c-api`, `c-api`, `bench` (`bench-rsf`, `bench-matmul`, `bench-tensor-ops`, `bench-sfd` via the `deps` module rooted at `src/_bench_deps.zig`), `inference-server`, `distributed-futhark` (requires `-Dgpu=true`), `pretokenize`, and the RTL steps `rtl`, `rtl-verilog`, `test-rtl`. Options: `-Dgpu`, `-Drtl`, `-Dskip-futhark`, plus the out-of-scope `-Dzk` / `-Dverify` and their steps `zk`, `test-zk`, `verify` — those two options and three steps stay exactly as they are, unused by this upgrade.
- Named build modules `core_relational`, `tensor_core_matmul`, and `tokenizer` are registered and passed to artifacts but **no source file imports them by name** (verified by grep: zero occurrences of `@import("core_relational")`, `@import("tensor_core_matmul")`, `@import("tokenizer")` in `src/`). All real imports are relative. These registrations are dead configuration.
- `src/hw/rtl/rtl_sim_main.zig` is rooted at `src/hw/rtl/` and imports `../../index/ssi.zig`, which escapes its module root; the `-Drtl` path is therefore only compilable after the build-graph correction in Phase 0.

Model dimensions (verified):

- `src/main_distributed_futhark.zig` reads `JAIDE_MODEL_DIM` (default **98304**) and `JAIDE_LAYERS` (default **24**).
- `model_dim` is the **full latent row length**; it must be even (`TrainerError.InvalidModelDim` / `AccelError.InvalidDimensions` otherwise). `half = model_dim / 2` is the RSF coupling width `dim`: coupling weights are `[dim, 2]` per layer per branch (`accel.rsf_coupling_width = 2`), and a latent row is `2·dim = model_dim` floats. For the default configuration `dim = half = 49152` and `D = model_dim = 98304 = 3 · 2^15`.

RSF architecture (`src/processor/rsf.zig`, 2141 lines):

- `RSFLayerConfig { clip_min: f32 = -5.0, clip_max: f32 = 5.0, seed_offset: u64 = 0, grad_mean: bool = true }`; `RSFConfig { clip_min, clip_max, grad_mean, max_dim = 1<<20, max_layers = 1<<20 }`.
- `LayerCore` holds `s_weight`, `t_weight: Tensor` of shape `[dim, 2]` where column 0 is the weight and column 1 is the bias, optional `s_weight_grad`/`t_weight_grad`, `dim`, clip bounds, an `rwlock`, and per-layer seed handling. Constants: `LAYER_TARGET_SPECTRAL_NORM = 0.9`, `LAYER_SPECTRAL_POWER_ITERATIONS = 30`, `SAVE_VERSION = 6`.
- Coupling algebra (the exact math that must be preserved and become the single source of truth):
  - State row layout: `row[0..dim] = x1`, `row[dim..2*dim] = x2`.
  - Scale path: `pre_s[d] = s_w[d] * x2[d] + s_b[d]`; `c[d] = clamp(pre_s[d], clip_min, clip_max)`; `scale[d] = exp(c[d])`; log-det contribution of the row is `sum_d c[d]`.
  - Forward: `x1'[d] = x1[d] * scale[d]`; `trans[d] = t_w[d] * x1'[d] + t_b[d]`; `x2'[d] = x2[d] + trans[d]`.
  - Inverse (exact): `trans[d] = t_w[d] * y1[d] + t_b[d]`; `y2[d] -= trans[d]`; recompute `scale` from the recovered `y2`; `y1[d] /= scale[d]`.
  - Backward (adjoint, from saved inputs): `dy1_total[d] = dy1[d] + t_w[d] * dy2[d]`; `dx1[d] = dy1_total[d] * scale[d]`; `ds[d] = 0` if `pre_s` saturated, else `dy1_total[d] * (x1[d] * scale[d]) + logdet_adjoint`; gradient accumulation with `grad_scale`: `∂t_w += dy2 * grad_scale * (x1 * scale)`, `∂t_b += dy2 * grad_scale`, `∂s_w += ds * grad_scale * x2`, `∂s_b += ds * grad_scale`; `dx2[d] = dy2[d] + s_w[d] * ds[d]`.
  - After every weight mutation, spectral norm is re-constrained to 0.9 by 30 power iterations (`constrainSpectralNorm` / `spectralNormPowerIteration`).
- `RSFCore` owns the layer array, `RSFConfig`, an `OFTB` instance applied after each layer's coupling in forward and before it in inverse, an `RSFAccelerator` GPU handle with FP16 weight upload and CPU cross-validation (`MODEL_CROSS_CHECK_ABS_TOL/REL_TOL = 5e-2`, `validateAcceleratorAgainstCPU` at line 1035), and weight-version tracking for GPU resync.
- Handle-based global registries: `RSF { id, ctrl }` with refcounted `acquire`/`release`/`requestDestroy` plus `shutdownGlobalRegistries()`; per-layer `RSFLayer` follows the same pattern.
- Public `RSF` API (to preserve unless this prompt says otherwise): `init`, `initWithConfig`, `deinit`, `isGPUAvailable`, `syncWeightsToGPU`, `ensureGradients`, `zeroGradients`, `gradientL2Norm`, `applyGradientStep`, `forwardCPU`, `forward`, `inverse`, `backward`, `forwardWithLogDet`, `meanLogDetJacobian`, `backwardWithLogDet`, `notifyWeightsChanged`, `verifyInvertible`, `save`, `load`, `loadWithConfig`, `saveLoadRoundtrip`.
- Save format: `SAVE_VERSION 6`, CRC32-protected, temp-file + atomic rename; `loadWithConfig` validates and applies a policy config.

OFTB (`src/processor/oftb.zig`, 293 lines):

- `OFTB { dim }` where `dim` is the half-width; slices are `2·dim` long. `FRACTAL_SCALE = 0.7071067811865476`, `FRACTAL_SCALE_SQ = 0.5000000000000001`, `LOG_DET_JACOBIAN = 0.0`.
- `vectorLen()` returns 16 when `x86_64` with `avx512f`, else 8; every kernel is a `@Vector(VLEN, f32)` loop plus scalar tail.
- `forwardSliceInPlace`: `x1' = (x1 − x2)·c`, `x2' = (x1 + x2)·c`. `backwardSliceInPlace` (also used as `inverseSliceInPlace` and as the gradient adjoint): `g1' = (g1 + g2)·c`, `g2' = (g2 − g1)·c`. `forwardInPlace`/`backwardInPlace`/`inverseInPlace` are the `Tensor`-typed wrappers; `forwardBackwardFusedInPlace` and `symplecticReversalInPlace` combine activation and gradient passes.
- The forward slice map is a rotation by 45° in each `(x1[d], x2[d])` plane; its 8th power is the identity (`R^8 = I`), its determinant is 1, and its adjoint equals its inverse.

SFD (`src/optimizer/sfd.zig`, 1416 lines):

- Contains a **private duplicate tensor stack** (`Tensor`, `Shape`, `TensorFlags`, `Precision`, `erfApprox`, `quantizeValue`) plus bridges `fromCoreTensor`/`toCoreTensor` to `src/core/tensor.zig`.
- `SFDConfig { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, clip_threshold = 0.1, weight_floor = 1e-3, fisher_max = 1e6, warmup_steps = 10, use_external_fisher = false }`.
- `SFD` is a flat-vector optimizer: `init(allocator, param_size)`, state = `fisher_diag`, `momentum_buffer` over `param_size` elements. Per-element semantics (the momentum/clip/skip structure must be preserved; the diagonal preconditioner is replaced — see Section 4.6): momentum `m ← β1·m + (1−β1)·g` (non-finite `g` treated as 0 for momentum, old momentum kept); Fisher `F ← β2·F + (1−β2)·g²` capped at `fisher_max`; bias corrections `m_correction`, `f_correction`; warmup factor; `delta = lr_eff · m̂ / (√F̂ + eps)`; `parameter_scale = max(|θ|, weight_floor)`; `delta` clamped to `±clip_threshold · parameter_scale`; `θ ← θ − delta` only when finite.
- Also contains: `KFACBlock` (generic input/output-dim Kronecker diagonal blocks, `updateStatistics`, `preconditionGradient`), `SpectralNormalizer` (power-iteration weight normalization, `lipschitzRegularization`), `adaptiveLR`, `spectralClip`, `accumulateFisher`, `clipGradNorm`, `ampSchedule` (cosine), `saveState`/`loadState` (magic `0x53464433`, flat format), `writeBackFP16`/`loadFromFP16`, `warmStart`, `varianceReduction`.
- Production usage: `src/distributed/distributed_trainer_futhark.zig` uses only `sfd.SpectralNormalizer`; it implements RSF-layer Fisher diagonals (`fisher_s`/`fisher_t`, gamma default 0.99, epsilon 1e-8, per-layer `[dim, 2]` arrays, master weights, momentum) itself, and executes GPU SFD updates through `src/hw/accel/accel_interface.zig`. `src/core/model_io.zig` re-exports `saveSFDState`/`loadSFDState`.

GPU / Futhark acceleration stack (verified):

- `src/hw/accel/main.fut` (497 lines): `oftb_scale_f32 = 0.7071067811865476`; `rsf_stack_coupling_row [half]` (coupling, then the OFTB rotation `(o1, o2) = ((y1−y2)c, (y1+y2)c)`, then `clamp_f16_value`); `rsf_stack_invert_row [half]` (OFTB adjoint `(u1, u2) = ((y1p+y2p)c, (y2p−y1p)c)`, then the exact coupling inverse); `sfd_fisher_update_core [d][e]` (diagonal Fisher EMA capped at `1e6`, bias corrections via `f32.** step_f`, `raw_step = lr · m̂ / (√max(F̂,0) + eps)`, trust clip `±trust_ratio · max(weight_floor, |w|)`, `sanitize_f32` on every intermediate). Entry points: `matmul`, `rsf_forward`, `master_weights_to_f16_3d`, `stack_update_sfd_master`, `master_weights_to_f16_2d`, `embedding_update_sfd_master`, `scale_matrix_f32`, `clip_matrix_global_norm_f32`, `embedding_forward_padded`, `embedding_backward_padded`, `stack_spectral_normalize` (takes `power_iters: i64`, calls `spectral_normalize_matrix`, which runs a `loop (u, v) … for iteration < max(1, power_iters)` power iteration), `embedding_spectral_normalize` (dense `[vocab_size][dim]` with ping-pong `u`/`v`), `graph_batch_encode`, `embedding_sum_squares`, `rsf_stack_forward`, `rsf_stack_inverse`, `rsf_stack_backward_gradients_fused`.
- `rsf_stack_backward_gradients_fused [batch_size][seq_len][half][num_layers]` takes `(final_outputs, targets, originals, lengths, weights_s, weights_t, grad_mean, gradient_scale, clip_min, clip_max, reconstruction_alpha, forward_scale, logdet_weight)` and returns a 6-tuple `(grad_s: *[num_layers][half][2]f32, grad_t: *[num_layers][half][2]f32, input_delta: *[batch_size][seq_len][half*2]f16, loss: f32, recon_loss: f32, logdet_mean: f32)`. It walks layers in reverse from the final outputs using the invert-row math, masks gradients where `|y| ≥ 60000`, computes `dy1_total`, `x2`, `pre_scale`, `scale`, `x1`, `dx1`, `ds` (zeroed outside `[clip_min, clip_max]`, shifted by `ld_shift = logdet_weight / valid_tokens`), `dx2`, accumulates `gs_l_total = [Σ ds·x2, Σ ds]` and `gt_l_total = [Σ h2·u1, Σ h2]` per layer, and returns `loss` (MSE over `count_elements`), `recon_loss` (MSE of the reconstructed trajectory against `originals`), and `logdet_mean = Σ_t Σ_d clipped / valid_tokens`.
- `src/hw/accel/futhark_kernels.fut` (1292 lines): `topk` (radix sort by `f32_total_order`), `rsf_scatter` (permutation-index mixing with `inv_sqrt2`), `rsf_flow`, `rsf_flow_logdet`, `rsf_invert_flow`, `rsf_forward_layer`, `rsf_forward_multi`, `rsf_backward_scatter`, `rsf_backward_flow`, entries `rsf_forward_multilayer`, `clip_fisher` (= `spectral_clip`), `update_fisher` (= `fisher_diagonal_update`), `compute_natural_grad` (= `spectral_natural_gradient`), `sfd_fused_step_1d`, and the `rgpu_*` family (`rgpu_compute_fractal_dimension`, complex `rgpu_hadamard_transform`, `rgpu_fractal_transform`, `rgpu_hadamard`, `rgpu_hadamard_batch`, `rgpu_fractal_xform`). The `rsf_flow`/`rsf_scatter` family is a permutation-parameterized variant of the coupling used by `rsf_forward_multilayer`; it does not apply the OFTB rotation and is not on the stack training path.
- `src/hw/accel/futhark_bindings.zig` (449 lines): opaque handles `struct_futhark_context`, `struct_futhark_f16_2d/3d`, `f32_1d/2d/3d`, `u64_1d`, `i64_1d`, and tuples `tup5_embedding_spectral`, `tup3_stack_spectral`, `tup7_graph_encode`, `tup3_arr2d_f32_arr2d_f32_arr2d_f32`, `tup6_fused_stack_gradients`, `tup3_stack_sfd`; raw accessors `futhark_values_raw_f32_2d/3d`; `futhark_entry_stack_spectral_normalize(ctx, out, weights, target, power_iters)`.
- `src/hw/accel/accel_interface.zig`: `RSFAccelerator` (`model_dim`, `num_layers`, `half = model_dim / 2`), `RSFOptimizerState { master_weights_s/t, momentum_s/t, fisher_s/t: []f32, step: u64 }` with every array of length `num_layers·half·2`, `setOptimizerState(master_weights_s, master_weights_t, momentum_s, momentum_t, fisher_s, fisher_t, step)` (line 2991), `applyStackGradientsSFD`, `applyGradientsSFD`, `applyUpdateFusedSFD`, `spectralNormalizeLayers(target, iterations)` (line 2830), `fusedTrainingStep(inputs, targets, sequence_lengths, grad_mean, gradient_scale, reconstruction_alpha, forward_scale, logdet_weight) -> FusedStepResult` (line 2515), `FusedStepScalars { loss, reconstruction_loss, logdet_mean }`, `forwardFromTensor` (requires `cols == model_dim`).

Relational core and trainer (verified):

- `src/core_relational/nsir_core.zig`: `bitmaskWordCount(node_count)` (line 493) and `bitmaskSignalPropagate(bitmask: []const u64, node_count: usize, signal: []const f32, decay: f32, out: []f32) void` (line 497) — a scalar implementation with early `return` on bad inputs (leaving `out` unwritten) and per-neighbor branching; plus `bitmaskIntersectionCount`, `bitmaskUnionInPlace`, `bitmaskDensity`, and `SelfSimilarRelationalGraph.exportAdjacencyBitmask` (line 1497). `src/core_relational/c_api.zig` imports `nsir_core.zig` and re-exports its graph and `Qubit` symbols, so the existing `bitmaskSignalPropagate` signature must keep compiling untouched for the C ABI.
- `src/distributed/distributed_trainer_futhark.zig`: `CHECKPOINT_VERSION = checkpoint_envelope.VERSION = 7` (`src/distributed/checkpoint_envelope.zig`); `trainPreparedStepFuthark` (line 1587) prepares `[batch][seq][model_dim]` FP16 input/target tensors and calls the fused step; `applyEmbeddingSpectralNormalization` (line 2554) power-iterates the dense `[vocab_size][dim]` embedding; the checkpoint writes `snapshot.rsf_optimizer_state.fisher_s` and `.fisher_t` as flat f32 arrays (lines 2157–2158) and reads them back at lines 2313–2316. `TrainerConfig` defaults: `reconstruction_alpha = 0.3`, `logdet_weight = fused_logdet_weight_default = -1e-3`, plus `spectral_target_norm`, `spectral_interval`, `trust_ratio`, `weight_floor`, `fisher_gamma = 0.99`, `fisher_epsilon ≥ 1e-12`, `clip_min`/`clip_max`.

Toolchain and build status (verified by execution in the target
environment — these results define what Phase 0 must confirm and what it
must not assume):

- Zig 0.14.1 is installed via `python3 -m venv /home/user/venv &&
  /home/user/venv/bin/pip install ziglang==0.14.1`; the binary is
  `/home/user/venv/bin/zig` (`zig version` → `0.14.1`), which satisfies
  `build.zig.zon`'s `minimum_zig_version = 0.14.1`.
- Futhark is NOT installed, and it cannot be installed from this sandbox right
  now: the GitHub release-asset CDN (`release-assets.githubusercontent.com`)
  fails with `SSL_ERROR_SYSCALL` for `curl` and `wget` alike, and
  `futhark-lang.org` and `hackage.haskell.org` are unreachable; no GHC, cabal,
  or stack is present to build the 0.26.4 source (which `codeload.github.com`
  does serve), and `src/hw/accel/futhark.pkg` additionally needs `futhark pkg
  sync` for its `sorts 0.6.1` dependency. The owner-supplied route remains
  `curl -LO https://github.com/diku-dk/futhark/releases/download/v0.26.4/futhark-0.26.4-linux-x86_64.tar.xz`,
  sha256 `bd07eba4c8f2b39ed7b494bf880a3fc5fe46254cd0cabd8ca1a63e5cf240f300`,
  `tar xf … && cd futhark-0.26.4 && PREFIX=/home/user/.local make install`.
  Treat Futhark availability as a blocker to re-test at Phase 0, never as an
  assumption.
- The generated accelerator sources `src/hw/accel/{main_cpu,main_gpu,futhark_kernels}.{c,h}`
  are gitignored and absent, so `-Dskip-futhark=true` fails with `FileNotFound`
  on `src/hw/accel/main_cpu.c` until `zig build regen-futhark` has produced
  them — and that step itself needs the `futhark` binary. This is the single
  hard external dependency of the entire build.
- Verified PASSING today with no Futhark and no build system at all — plain
  `zig test src/<wrapper>` for 18 of the 19 registered specs, 562 tests, zero
  failures:

  | `build.zig` step | wrapper | verified result |
  | --- | --- | --- |
  | `test-tensor` | `src/test_root_tensor.zig` | 29 passed |
  | `test-memory` | `src/test_root_memory.zig` | 19 passed |
  | `test-gpu-memory` | `src/test_root_gpu_memory.zig` | 8 passed |
  | `test-sfd` | `src/test_root_sfd.zig` | 52 passed |
  | `test-embedding` | `src/test_root_embedding.zig` | 32 passed |
  | `test-oftb` | `src/test_root_oftb.zig` | 32 passed |
  | `test-nsir` | `src/test_root_nsir.zig` | 11 passed |
  | `test-reasoning` | `src/test_root_reasoning.zig` | 67 passed |
  | `test-crev` | `src/test_root_crev.zig` | 48 passed |
  | `test-surprise` | `src/test_root_surprise.zig` | 31 passed |
  | `test-temporal` | `src/test_root_temporal.zig` | 39 passed |
  | `test-vpu` | `src/test_root_vpu.zig` | 23 passed |
  | `test-fnds` | `src/test_root_fnds.zig` | 6 passed |
  | `test-formal` | `src/test_root_formal.zig` | 27 passed |
  | `test-security` | `src/test_root_security.zig` | 38 passed |
  | `test-quantum-adapter` | `src/test_root_quantum_adapter.zig` | 32 passed |
  | `test-signal` | `src/test_root_signal.zig` | 36 passed |
  | `stress-refcount` | `src/test_root_stress_refcount.zig` | 32 passed |

  Also passing without Futhark: `zig build check` (semantic analysis of every
  object, no linking, ~17 s), `zig build c-api`, `zig build test-c-api`
  (180 passed / 0 failed).
- Verified BLOCKED today: `zig build test-all` and every artifact built through
  the spec loop in `build.zig`, because line 258 calls `applyAccel` on all 19
  test artifacts and `applyAccel` does `if (cfg.codegen) |step|
  artifact.step.dependOn(step)` — `codegen` being the `run futhark` step. The
  same blocks `bench`, `inference-server`, and `distributed-server`.
  `-Dskip-futhark=true` is not a workaround: it only skips regeneration and then
  fails with `FileNotFound` on the gitignored `src/hw/accel/main_cpu.c`.
- Exactly one spec has a genuine accelerator dependency: `test-rsf`
  (`src/test_root_rsf.zig` → `src/processor/rsf.zig` →
  `src/hw/accel/accel_interface.zig`, whose line 7 reads
  `@import("build_options").gpu_acceleration` at container scope). Verified by
  supplying a hand-written options module on the command line —
  `zig test --dep build_options -Mroot=src/test_root_rsf.zig
  -Mbuild_options=<file defining gpu_acceleration, zk_enabled, verify_enabled,
  rtl_enabled> -lc` — after which it passes semantic analysis and fails at link
  time with undefined symbols `futhark_new_u64_1d`, `futhark_values_i64_1d`,
  `futhark_values_u64_1d`, `futhark_project_opaque_tup7_arr1d_u64_…` referenced
  from `accel_interface.zig` lines 998, 4537, 4564, 4569. Those symbols exist
  only in the generated `src/hw/accel/main_cpu.c`. So `test-rsf` requires both
  the build-provided `build_options` module and `zig build regen-futhark`.
  `test-gpu-memory` imports `hw/accel/gpu_memory.zig` and
  `hw/accel/compact_batch.zig` but never reaches a Futhark extern, so it links
  and passes without the accelerator.
- Out of scope and deliberately untouched: `src/verification/Verification.lean`,
  `src/zk/inference_trace.circom`, and the `build.zig` options `-Dverify` /
  `-Dzk` that drive them. No work, no requirements, no builds for them; do not
  remove or break their existing registration either. The Zig-side formal module
  `src/core_relational/formal_verification.zig` (2894 lines; imports
  `nsir_core.SelfSimilarRelationalGraph`, `Node`, `Edge`, `EdgeQuality`,
  `EdgeKey`; exercised by `test-formal`, 27 tests) IS in scope and must keep
  compiling and passing through the Phase 11 rewrite of `nsir_core.zig`.
- Rooting rule for standalone test runs: `zig test src/processor/oftb.zig` fails
  with "import of file outside module path" because that file imports
  `../core/tensor.zig`; root such runs at `src/` (as `src/test_root_*.zig` do)
  or pass explicit `-Mroot=…` module flags.

Core tensor (`src/core/tensor.zig`, 2629 lines):

- `Shape` (≤ 8 dims, strides, overflow-checked), `Tensor { data: []align(32) f32, base_data, shape, allocator, refcount: *usize, cow: *bool, huge_allocator_owner }` with COW-on-write refcounting, `initHuge` 2 MiB/1 GiB huge-page paths, `TensorIterator`, blocked GEMM (`NC = 4096`, `KC = 256`, `MC = 256`, `NR = 8`, `NR_AVX512 = 16`, pack `packA`/`packB`, `@Vector(8, f32)` SIMD with AVX-512 detection), `MatmulComptime`, plus helpers (`randomUniform`, `zeros`, `eye`, `clone`, `copyFrom`, `mulScalar`, `add`, `sub`, `normL2`, `spectralNorm`, `matmul`, `cholesky`, `outerProduct`, `save`/`load`). No RSF awareness today.
- `src/core/types.zig` (1987 lines) is the shared types module (`Error`, `BitSet`, `PRNG`, `RankedSegment`, `Fixed32_32`, `ContextWindow`, …) and already re-exports `Tensor`.

SSI (`src/index/ssi.zig`, 951 lines): height-capped (`max_height = 6`, `bucket_width = 6`) hierarchical index over token `Segment`s (`tokens`, `position`, `score`, `anchor_hash`, 64-bit MinHash `signature`), with `addSequence`, `retrieveTopK`, `compact`, `updateScore`, `getSegment`, `serialize`/`deserialize`, `exportToTensor`/`importFromTensor`, `merge`, `split`, `balance`, `stats`, `validate`. Consumers: `ranker.zig`, `model_io.zig`, `inference_server.zig`, `rtl_sim_main.zig`.

Ranker (`src/ranker/ranker.zig`, 1528 lines): `RankerConfig` constants, n-gram weights, LSH hash parameters, MinHash signatures and Jaccard estimators, `scoreSequence`, `scoreSequenceWithQuery`, `rankCandidates`, `rankCandidatesWithQuery`, `batchScore`, `topKHeap`, `updateWeights`, `minHash*`, `vectorScore`/`dotProductScore` (generic tensor dot products), `weightedAverage`, `exponentialDecay`, `normalizeScores`, `rankByMultipleCriteria`, `streamingRank`, `parallelScore`, `calibrateWeights` (hand-rolled SGD), `exportModel`/`importModel`.

VPU (`src/core_relational/vpu.zig`, 2468 lines): `BitmaskMatrix` (128-byte-aligned adjacency bitmasks over NSIR graphs, `booleanPropagate`, `signalPropagate` with decay, `andPopCount*`, `predecessorPopCounts`), `SimdVector(T, N)` generic SIMD type library (`F32x4`…`I32x8`), `VectorBatch`, `Matrix4x4`/`MatrixOps`, `RelationalVectorOps`, `MemorySlice`/`MemoryPool`, `VectorCache`, `VPUStatistics`, `VPU` facade (`processVectors`, `batchMatmul`, `computeGraphEmbeddings`, `quantumVectorOps`, similarity matrix, power iteration, bitmask helpers), `LNSValue`/`LNSInstruction`. No RSF involvement.

R-GPU (`src/core_relational/r_gpu.zig`, 2721 lines): `P2PTransferManager` (real CUDA runtime via `dlopen`, device discovery, NVLink5 mesh detection, peer access, device shard staging, async streams), `CoreState`/`MessageType`/`ProcessingCore`/`NoCMessage`/`RouteKey`/`AsynchronousNoC`, `GraphIsomorphismProcessor`, `DynamicEdgeWeighting`, `SparseActivationManager`, `PowerGatingController`, `RPGUStatistics`, and the `RelationalGraphProcessingUnit` facade operating on NSIR `SelfSimilarRelationalGraph`s. No RSF involvement.

Other dependents that Phase 12 must migrate: `src/core/model_io.zig` (`JAIDE40\x00` envelope, `CURRENT_VERSION = 1`), `src/api/c_api.zig`, `src/api/inference_server.zig`, `src/distributed/comm_backend.zig`/`comm_backend_cpu.zig` (both duplicate the operation-type tag list), `src/core_relational/dataset_obfuscation.zig` (its `OperationType` enum must gain the new native operation tags), `src/_bench_deps.zig`, and the `semantic_check_obj`/`distributed_check_obj` objects that make `zig build check` link-free.

## 3. Global constraints (non-negotiable)

1. **Toolchain and verified environment state**: Zig 0.14.1 is installed at `/home/user/venv/bin/zig` (from `pip install ziglang==0.14.1` inside `/home/user/venv`) and satisfies `build.zig.zon`. Futhark is absent and currently uninstallable from this sandbox (release CDN and `futhark-lang.org` unreachable; no GHC/cabal); the required route once the network allows it is the v0.26.4 tarball (sha256 `bd07eba4c8f2b39ed7b494bf880a3fc5fe46254cd0cabd8ca1a63e5cf240f300`) installed with `PREFIX=/home/user/.local make install`, followed by `zig build regen-futhark` to generate the gitignored `src/hw/accel/*.c|h` sources. You must compile, run, and iterate until the acceptance gates pass, and you must report per gate whether it passed, failed, or was blocked by the missing `futhark` binary — a blocked gate is recorded as blocked, never as passed, and never as a reason to write untested code. No dependency additions. Futhark/CUDA/GHC/Clash remain optional toolchains. Out of scope entirely: Lean formal verification (`src/verification/Verification.lean`, `-Dverify`) and Circom circuits (`src/zk/inference_trace.circom`, `-Dzk`) — produce no artifacts, requirements, proofs, or circuits for them, and leave their existing build registration untouched.
2. **Complete code**: deliver full file contents for every modified or created file. Never elide, abbreviate, or annotate code as "unchanged". Never provide pseudo-code.
3. **No placeholders**: no mocks, stubs, dummies, simulated results, fake statistics, hardcoded return values standing in for computation, `TODO`/`FIXME`/`unimplemented` markers, or `@panic` used to excuse missing implementations. The R-GPU is a software execution substrate by design — all of its paths must really execute RSF computations and report genuinely measured statistics.
4. **Invariant paradigm**: no perceptrons, no dense learned linear projections, no convolutions, no recurrent units, no attention (no softmax, no `QK^T`, no Q/K/V projections) may be introduced anywhere. The only learned parameters in the system are the per-channel rank-2 coupling pairs `(s_w[d], s_b[d])` and `(t_w[d], t_b[d])`. Every mixing operation across coordinates must be a fixed, unparameterized, orthogonal (or symplectic) transform — the OFTB rotation and the `Q_r ⊗ H_{2^k}` diffusion — or the coupling itself. The token embedding table is exempt and must remain a row lookup indexed by token id (Section 1, gate 15); it is never to be converted into, or replaced by, a dense learned projection.
5. **Bijectivity and exact volume tracking**: every map delivered must be exactly invertible in real arithmetic, and its log-det must be computed analytically from the clip values (`Σ_d c[d]` per row) or proven to be exactly zero (orthogonal factors). No determinant estimates, no Jacobian approximations, no numerical log-det via finite differences in production paths (finite differences are allowed inside tests only, as a cross-check).
6. **Error handling**: library code returns error unions; use `errdefer` for every fallible allocation; no `try`-free swallowing; no panics or `unreachable` reachable from valid input; validate all public inputs (dimensions, finiteness, shapes, bindings, mask triangularity) and return typed errors.
7. **Memory and concurrency**: preserve the existing allocator discipline (caller-supplied `Allocator`, no hidden global allocators beyond the existing registries), the refcount/COW tensor contract, the global handle registries with their rwlocks and `shutdownGlobalRegistries()`, and the lock ordering rules already present in `rsf.zig`. Every new shared structure is either immutable, lock-guarded, or single-threaded by contract — state which in the doc comment. Auxiliary activation memory in the new sequence and midpoint paths must be `O(1)` per row (Section 4.8, Section 4.7), never `O(seq_len²)` or `O(layers)` scratch per token.
8. **Numerics**: all float math in the new coupling kernels must match the reference scalar semantics defined in Section 2 within tolerance (abs 1e-5 / rel 1e-4 for forward/inverse vs the current scalar path; exact structural semantics for saturation flags). Use f32 throughout the hot paths; f64 only for accumulators (norms, statistics, spectral Gram sums) as specified per function. CPU and GPU paths must implement **the same** layer map — including the diffusion factor and the causal coupling mode — or `validateAcceleratorAgainstCPU` (tolerance 5e-2 on FP16 device arrays) will fail; any tolerance change must be justified in writing, not silently widened.
9. **Style**: match the repository's existing conventions — file-top imports, `const Allocator = std.mem.Allocator`, snake_case, doc comments on public declarations, tests colocated in the module file and pulled in by `src/test_root_*.zig` wrappers, build steps registered in `build.zig`.
10. **Behavioral preservation**: the C API ABI must not change; `src/core_relational/c_api.zig` must compile untouched. Where a numeric format changes (RSF save version, SFD state magic, trainer checkpoint version, SSI serialization version, ranker export version), implement a legacy reader, promote legacy state deterministically as specified, and prove it with a Phase 0 fixture round-trip test. Legacy RSF v6 models must reproduce their recorded outputs exactly, which requires the diffusion factor to be disabled for models loaded from v6 (Section 4.5, Phase 3).

## 4. Target architecture: the RSF substrate

### 4.0 Dependency hierarchy

The end state is a strict dependency hierarchy in which RSF is the foundation and the rewritten modules are RSF-native organs:

```
src/core/types.zig           (RSFSpace, RSFBinding, RSFDiffusionLayout,
                              RSFSequenceMask, existing types)          — no new deps
src/core/tensor.zig          (RSF-bound tensor + canonical coupling algebra
                              + exact rank-2 spectral norm + diffusion kernels
                              + causal bitmask coupling kernels:
                              the single source of truth)                — imports types
src/processor/oftb.zig       (orthogonal butterfly: rotation R + global
                              diffusion Θ = Q_r ⊗ H_{2^k})               — imports tensor, types
src/processor/rsf.zig        (model, registries, dispatch, latent states,
                              dual-frontier midpoint, sequence flow)     — imports oftb, tensor
src/optimizer/sfd.zig        (RSF-only optimizer, 2×2 block natural
                              gradient)                                  — imports rsf, tensor
src/index/ssi.zig            (latent index)                              — imports rsf, tensor, types
src/ranker/ranker.zig        (latent ranker with RSF head)               — imports rsf, ssi, sfd, tensor
src/core_relational/nsir_core.zig (bit-plane popcount propagation)       — no new deps
src/core_relational/vpu.zig  (RSF vector execution unit)                 — imports rsf, tensor, nsir_core
src/core_relational/r_gpu.zig (RSF sharded execution fabric)             — imports rsf, tensor, vpu, nsir_core
src/hw/accel/*.fut, futhark_bindings.zig, accel_interface.zig (device
                              implementations of the same layer map)     — host-side only
```

### 4.1 `src/core/types.zig` additions

```zig
pub const RSFSpace = enum(u8) {
    layer_weight_s,    // scale-coupling parameters, shape [dim, 2]
    layer_weight_t,    // translation-coupling parameters, shape [dim, 2]
    latent_state,      // RSF latent rows, shape [batch, dim, 2]
    gradient,          // coupling parameter gradients, shape [dim, 2]
    fisher,            // diagonal Fisher / Gauss-Newton statistics, shape [dim, 2]
    fisher_block,      // analytic 2x2 Fisher covariance blocks (F_ww, F_wb, F_bb), shape [dim, 3]
    momentum,          // optimizer momentum state, shape [dim, 2]
    master_weight,     // FP32 master copies behind FP16 device weights, shape [dim, 2]
    ranker_head,       // ranker head coupling parameters, shape [dim, 2]
    index_payload,     // SSI latent payloads, length 2*dim
};

pub const RSFBinding = struct {
    space: RSFSpace,
    model_id: u64,          // RSFCore registry id of the owning model (0 for detached payloads)
    layer_index: ?usize,    // null for non-layer spaces
    dim: usize,             // half-dimension (number of coupling cells per half)
};

pub const RSFBindingError = error{
    RSFBindingRequired,
    RSFSpaceMismatch,
    RSFModelMismatch,
    RSFLayerMismatch,
    RSFDimMismatch,
};

/// Block decomposition of a latent row of length `row_len` for the global
/// diffusion operator Θ = Q_r ⊗ H_{2^k}: row_len = radix * block, block = 2^stages.
pub const RSFDiffusionLayout = struct {
    row_len: usize,
    radix: usize,     // r, the odd part of row_len (r >= 1)
    block: usize,     // 2^k, the power-of-two block length (block >= 2 always, since row_len is even)
    stages: usize,    // k = log2(block), the number of butterfly stages per block
};

/// Returns null only when row_len == 0. A latent row length is always even
/// (row_len = 2*dim), so block >= 2 and the decomposition always exists.
pub fn rsfDiffusionLayout(row_len: usize) ?RSFDiffusionLayout;

/// Strictly lower-triangular binary causal mask over a token sequence.
pub const RSFSequenceMask = struct {
    seq_len: usize,
    words_per_row: usize,   // ceil(seq_len / 64), same convention as nsir_core.bitmaskWordCount
    words: []u64,           // packed row-major bits; row i is words[i*words_per_row ..][0..words_per_row]
    nnz: usize,             // number of set bits, computed once at init (drives the complexity choice in Phase 1)
    full_causal: bool,      // true iff every j < i is set: enables the O(seq_len*dim) running-accumulator path

    /// `bytes` is row-major [seq_len][seq_len] with values restricted to {0,1}. init validates
    /// bytes.len == seq_len*seq_len, every value in {0,1}, and C[i][j] == 0 for all j >= i;
    /// any violation returns error.InvalidCausalMask.
    pub fn initFromBytes(allocator: Allocator, seq_len: usize, bytes: []const u8) !RSFSequenceMask;
    pub fn initCausal(allocator: Allocator, seq_len: usize) !RSFSequenceMask; // every j < i set; full_causal = true
    pub fn initZero(allocator: Allocator, seq_len: usize) !RSFSequenceMask;   // no coupling; must reproduce the per-token path
    pub fn get(self: *const RSFSequenceMask, i: usize, j: usize) bool;
    pub fn density(self: *const RSFSequenceMask) f32;  // nnz / (seq_len*(seq_len-1)/2); 0 for seq_len < 2
    /// Row-major [seq_len][seq_len] u8 form required by the Futhark entries (Phase 5) — allocated on demand,
    /// never held: the packed form is the storage of record (seq_len^2/8 bytes, not seq_len^2).
    pub fn toBytes(self: *const RSFSequenceMask, allocator: Allocator) ![]u8;
    /// Visits the set bits of row i in ascending j order (sparse accumulation path of Phase 1).
    pub fn rowSetBits(self: *const RSFSequenceMask, i: usize, ctx: anytype, comptime visit: fn (@TypeOf(ctx), usize) void) void;
    pub fn clone(self: *const RSFSequenceMask, allocator: Allocator) !RSFSequenceMask;
    pub fn deinit(self: *RSFSequenceMask) void;
};
```

Add these errors to the existing `types.Error` set (Zig error sets are open; existing error names must not change): `RSFBindingRequired`, `RSFSpaceMismatch`, `RSFModelMismatch`, `RSFLayerMismatch`, `RSFDimMismatch`, `InvalidCausalMask`, `InvalidDiffusionLayout`.

### 4.2 Canonical coupling algebra (single source of truth)

`src/core/tensor.zig` becomes the sole implementation of the coupling algebra from Section 2, as vectorized kernels over strided `[dim, 2]` parameter tensors and row states. `rsf.zig`, `oftb.zig`, `vpu.zig`, and `r_gpu.zig` must call these kernels (or `comptime`-instantiate them); duplicating the formulas anywhere else is a defect. `sfd.zig`, `ssi.zig`, and `ranker.zig` operate on tensors produced/consumed by these kernels. The Futhark kernels in `src/hw/accel/` are the device transcription of the same formulas and must be kept term-for-term identical (Section 3, constraint 8).

### 4.3 RSF latent state

`src/processor/rsf.zig` exports the canonical state type that SSI, Ranker, VPU, and R-GPU exchange:

```zig
pub const RSFLatentState = struct {
    data: Tensor,        // shape [batch, dim, 2], binding .latent_state with the model id
    log_det: f32,        // accumulated log|det J| through the stack (mean over batch)
    pub fn init(allocator: Allocator, model: *const RSF, batch: usize) !RSFLatentState;
    pub fn fromHalves(allocator: Allocator, model: *const RSF, x1: *const Tensor, x2: *const Tensor) !RSFLatentState;
    pub fn forwardThrough(self: *RSFLatentState, model: *RSF) !void;   // CPU/GPU dispatch, log_det accumulation
    pub fn inverseThrough(self: *RSFLatentState, model: *RSF) !void;
    pub fn roundtripError(self: *const RSFLatentState, model: *RSF, allocator: Allocator) !f32; // relative L2 of inverse(forward(x)) - x
    pub fn evenView / oddView / clone / deinit ...
};
```

### 4.4 Mathematical identity of the model (normative documentation)

The implementation and its doc comments must state the model's identity explicitly, because every optimization below follows from it:

- **JAIDE is an Incompressible Tame Diffeomorphic Manifold Integrator.** Each layer is a tame diffeomorphism of the latent space: it is smooth, bijective, has a smooth inverse computed in closed form, and its Jacobian determinant is bounded and tracked exactly (`exp(Σ_d c[d])` with `c[d] ∈ [clip_min, clip_max]`, hence `|det J| ∈ [exp(dim·clip_min), exp(dim·clip_max)]`).
- **The stack is an exact automorphism of the latent space**, `F ∈ Aut(AD)`: `F` is bijective, `F⁻¹` is computed by reversing the layer order and inverting each layer in closed form, and `F⁻¹ ∘ F = id` up to floating-point rounding only. This is the property that makes the dual-frontier midpoint scheme of Section 4.7 exact rather than approximate.
- **Liouville volume preservation of information.** Because every layer is a bijection of the latent, the mutual information between the input and any intermediate representation is invariant: `I(X; X^{(l)}) ≡ H(X)` for every layer index `l`. No information is destroyed by depth, therefore no LayerNorm, no residual skip connection, and no dropout is required or permitted, and the layer count is scalable (`L = 24` to `L = 1000`) without representational collapse. The numerically verifiable form of this claim is the roundtrip gate (`roundtripError ≤ 1e-4`) plus the additive exact log-det. Information preservation is a statement about bijectivity, **not** about conditioning: `scale[d] = exp(c[d])` with `c[d] ∈ [clip_min, clip_max]` can amplify a channel by up to `e^5 ≈ 148` per layer, so what limits depth in practice is finite-precision range, not lost information. Depth scaling is therefore *monitored*, never assumed: `RSF.stateGrowthReport` (Phase 3) reports `max|x|` at every 8-layer boundary, the device path keeps its `clamp_f16_value`/`±65504` guard, and the CPU path keeps `ensureFiniteSlice`. Any statement about `L = 1000` is an extrapolation of this monitored behavior and must be labeled as an extrapolation wherever it appears; the tested depth range is `L ∈ {8, 24, 64}` (Phase 13). Adding LayerNorm, residual skips, or dropout to "help" deep stacks is prohibited by constraint 4 of Section 3 — if a depth overflows, the measured limit is reported, not papered over.
- **Symplectic / orthogonal factor structure.** The layer map factors as `Θ ∘ R ∘ C` (Section 4.5) where `C` is the volume-changing coupling with analytic log-det, `R` is the OFTB rotation with `det R = 1` and `R^8 = I`, and `Θ` is the global diffusion with `|det Θ| = 1` and `Θ² = I`. Only `C` changes volume, and it does so by an exactly known factor.
- **C8 symplectic resonance.** Since `R^8 = I`, the orthogonal factor of every group of 8 layers is the identity, so 8 layers form one closed phase cycle of the linear part of the flow. Drift control therefore checks the phase cycle every 8 layers rather than back-propagating through all `L` layers: `RSF.resonanceDrift` (Phase 3) evaluates the Hamilton energy `E = ‖x1‖² + ‖x2‖²` at each 8-layer boundary and reports `max_l |E_l − E_0| / E_0`. The coupling factor is *not* of finite order, so `E` is a bounded diagnostic, not an exact invariant: the exact invariants are bijectivity, the additive analytic log-det, `R^8 = I`, and `Θ² = I`. The implementation and its tests must describe `E` in exactly these terms — do not assert energy conservation as an exact law.
- **Three-Mark training.** The trainer's fused triple-signal objective (prediction mismatch, reconstruction, log-det volume) is the discrete form of the Euler–Arnold–Liouville action integral for the flow, and training is the direct numerical solution of its minimum principle: `S[θ] = Σ_t [ ‖F_θ(x_t) − y*_t‖² + α·‖x̂_t − x_t‖² + λ·log|det J_{F_θ}| ]` with `α = reconstruction_alpha = 0.3` and `λ = logdet_weight = −1e-3` (the trainer's existing defaults). The dual-frontier midpoint scheme of Section 4.7 evaluates the same action principle with half the sequential depth by exploiting `F ∈ Aut(AD)`.

### 4.5 Canonical layer map with global cross-channel diffusion

The per-layer map becomes:

```
forward:  a  ──C──▶  y  ──R──▶  p  ──Θ──▶  q          (layer output q)
inverse:  q  ──Θ──▶  p  ──R^T──▶ y  ──C⁻¹──▶ a
```

- `C` = the coupling of Section 4.2, `log|det J_C| = Σ_d c[d]`.
- `R` = the existing OFTB rotation, `R^T = R⁻¹`, `det R = 1`, `R^8 = I`, `log|det J_R| = 0`.
- `Θ` = the **global diffusion operator**, unparameterized, `log|det J_Θ| = 0` exactly:

```
row_len = 2·dim = r · 2^k  with r odd  (r = 3, k = 15 for the default model_dim = 98304)
Θ = Q_r ⊗ H_{2^k}
```

  - `H_m` is the normalized Fast Walsh-Hadamard transform on `m = 2^k` elements, applied **within each of the `r` contiguous blocks**: butterfly stages `h = 1, 2, 4, …, m/2` with `u = (x_j + x_{j+h})/√2`, `v = (x_j − x_{j+h})/√2`. `H_m = H_m^T = H_m⁻¹`, `|det H_m| = 1`, cost `m·log2(m)/2` butterfly pairs.
  - `Q_r = I_r − (2/r)·11^T` is the `r`-point symmetric orthogonal involution applied **across blocks at fixed offset**: with `S[o] = Σ_{b<r} x[b·m + o]`, `Θ_Q(x)[b·m + o] = x[b·m + o] − (2/r)·S[o]`. `Q_r = Q_r^T = Q_r⁻¹`, eigenvalues `{−1, 1, …, 1}`, `det Q_r = −1`.
  - The two factors commute (`I_r ⊗ H_m` versus `Q_r ⊗ I_m`), so `Θ` is an exact involution: `Θ² = Q_r² ⊗ H_m² = I`, `Θ^T = Θ`, `|det Θ| = |det Q_r|^m · |det H_m|^r = 1`. Therefore **the same routine is the forward transform, the inverse transform, and the gradient adjoint** — one implementation, three uses. Total cost per row: `k` butterfly stages plus one `r`-way linear combination = `O(D log D)` with `D = row_len`.
- Diffusion is enabled per model by `RSFConfig.global_diffusion: bool` (default `true` for newly created models). A model loaded from `SAVE_VERSION 6` sets it to `false`, so v6 checkpoints reproduce their recorded outputs bit-for-bit (Section 3, constraint 10). `OFTB` carries the flag and applies `Θ` inside `forwardSliceInPlace` (after `R`) and inside `backwardSliceInPlace` (before `R^T`), which keeps `backwardSliceInPlace` simultaneously the inverse and the adjoint.
- With diffusion enabled, a single-coordinate perturbation reaches every block of the row after one layer: cross-channel isolation is eliminated without any learned dense matrix.

### 4.6 Analytic 2×2 block-diagonal Natural Gradient (SFD)

For each coupling row index `d` and each branch (`s` and `t` separately), the two parameters `θ_d = (w_d, b_d)` (column 0 = weight, column 1 = bias) are preconditioned by the **full 2×2 Fisher covariance block**, not by two independent diagonals:

```
F_ww,d ← β2·F_ww,d + (1−β2)·g_w,d²
F_wb,d ← β2·F_wb,d + (1−β2)·g_w,d·g_b,d
F_bb,d ← β2·F_bb,d + (1−β2)·g_b,d²
stored per d as a [dim, 3] tensor of space .fisher_block, capped at fisher_max per component
```

With bias corrections `m̂ = m/(1 − β1^t)`, `F̂ = F/(1 − β2^t)` and damping `λ = fisher_epsilon`, the inverse square root is computed in closed form by the Cayley–Hamilton resolvent for a 2×2 SPD matrix — no eigendecomposition, no iteration:

```
A = F̂_ww + λ,  B = F̂_wb,  C = F̂_bb + λ
det = max(1e-12, A·C − B²)
√det = sqrt(det)
S = sqrt(A + C + 2·√det)          // = √λ₁ + √λ₂  (both eigenvalues of F+λI are positive)
α = 1 / (√det · S)
(F + λI)^{-1/2} = α · [[C + √det, −B], [−B, A + √det]]
```

The preconditioned step is bounded by the existing trust ratio and applied to the master weights:

```
Δw = clip( α · ((C + √det)·m̂_w − B·m̂_b) · lr_eff,  −τ·max(|w|, weight_floor),  +τ·max(|w|, weight_floor) )
Δb = clip( α · (−B·m̂_w + (A + √det)·m̂_b) · lr_eff, −τ·max(|b|, weight_floor),  +τ·max(|b|, weight_floor) )
θ ← θ − Δ   (only when Δ is finite)
```

`lr_eff` includes the existing warmup factor; `τ = clip_threshold`. Derivation to record in the doc comment: for `M = F + λI` with eigenvalues `λ₁, λ₂`, `√det = √(λ₁λ₂)`, `S = √λ₁ + √λ₂`, and `M^{-1/2} = (M^{1/2})^{-1} = ((√λ₁+√λ₂)I − M)/√(λ₁λ₂)` expanded in the basis `{I, M}` — which is exactly the matrix above. The mandatory validation test asserts `‖M^{-1/2}·M·M^{-1/2} − I‖₂ < 1e-5` over a PRNG sweep of `(A, B, C)`.

**Why this can accelerate convergence, and by how much (normative).** Write `F̂_d + λI = [[A, B], [B, C]]` and `ρ_d = B/√(A·C)`. Diagonal preconditioning leaves the scaled curvature matrix `[[1, ρ], [ρ, 1]]`, whose condition number is `(1+|ρ|)/(1−|ρ|)`; the exact block inverse square root reduces it to `1`. The available gain is therefore governed by the measured off-diagonal correlation, so `StepStats` must report `fisher_offdiag_mean_abs` **and** `correlation_abs_mean` (mean `|ρ_d|` over rows and layers, f64) — the claim has to be checkable, not asserted. In the RSF coupling the two gradient components of row `d` are `g_w = Σ_b ds·x2[d]` and `g_b = Σ_b ds`, so `ρ_d` is exactly the batch correlation between `x2[d]` and the constant `1`: it is large whenever `x2[d]` has a non-zero mean, which is the normal case after the first layers, and it is near zero for a zero-mean latent — in which case the block preconditioner legitimately degenerates to the diagonal one and no speedup is to be claimed.

Storage and layout consequences (all three must change together, in Phases 4, 6, and 12): SFD holds `fisher_blocks_s`/`fisher_blocks_t: []Tensor` of shape `[dim, 3]`; `accel.RSFOptimizerState` and `setOptimizerState` carry `fisher_blocks_s`/`fisher_blocks_t` of length `num_layers·half·3`; the flat export layout becomes `master_s ‖ master_t ‖ momentum_s ‖ momentum_t ‖ fisher_blocks_s ‖ fisher_blocks_t`; and the trainer checkpoint version increments (`CHECKPOINT_VERSION` 7 → 8) with a legacy reader that promotes the v7 diagonal state **per branch, from that branch's own two columns**:

```
blocks_s[d*3+0] = fisher_s[d*2+0]   // F_ww of the s branch
blocks_s[d*3+1] = 0                 // F_wb — no cross term was ever stored
blocks_s[d*3+2] = fisher_s[d*2+1]   // F_bb of the s branch
blocks_t[d*3+0] = fisher_t[d*2+0] ;  blocks_t[d*3+1] = 0 ;  blocks_t[d*3+2] = fisher_t[d*2+1]
```

Do **not** cross the branches: `fisher_s` and `fisher_t` are the `[dim, 2]` states of two *different* coupling matrices, not the two entries of one 2×2 block. The promotion is exact for a diagonal-preconditioned history (`B = 0` ⇒ the block step reduces to the diagonal step, which is a mandatory Phase 4 test), and must be documented as such wherever it is implemented (`accel.promoteDiagonalFisher`, the trainer's v7 reader, `SFD.loadState`'s v1 path).

### 4.7 Dual-frontier symplectic midpoint collision (Smale midpoint shooting)

Because `F ∈ Aut(AD)`, the fitting condition `F_θ(x₀) = y*` is equivalent to a midpoint collision, and the two halves of the layer stack can be traversed **concurrently** from opposite ends:

```
M = L / 2  (integer division; layers 0..M−1 = forward frontier, layers M..L−1 = backward frontier)
forward frontier:   z_M = (Θ R C)_{M−1} ∘ … ∘ (Θ R C)_0 (x₀)         // M sequential steps
backward frontier:  w_M = (Θ R C)_M⁻¹ ∘ … ∘ (Θ R C)_{L−1}⁻¹ (y*)     // L−M sequential steps
collision loss:     ℓ = (1/(T·D)) · Σ_{t active} ‖z_{M,t} − w_{M,t}‖²
volume term:        logdet F = Σ_{l<M} Σ_d c_l[d]  +  Σ_{l≥M} Σ_d c_l[d]   (both accumulated as +Σ clip)
```

- Sequential depth per frontier is `⌈L/2⌉` instead of `L` — the 2× latency reduction of gate 9. The two frontiers are data-independent, so they are issued concurrently (two threads on CPU; one fused kernel launch on device).
- **Why `logdet_backward` is the log-det of `F_{M..L−1}` and not of its inverse.** During the inverse traversal each layer first recovers `x2` and then recomputes `s = clip(W_s·x2 + b_s)` from it, so the states it visits are exactly the states the forward traversal of those layers would visit; the accumulated `Σ_d c[d]` is therefore the layer's forward log-det. Both frontiers accumulate `+Σ clip`, and `logdet_total = logdet_forward + logdet_backward = log|det J_F|` — no sign flip, no double counting. State this in the doc comment of `midpointCollision` and assert it in Phase 13's `rsfNative_midpointEquivalence` (compare `logdet_total` against `forwardWithLogDet` on the same weights).
- The backward frontier uses the **inverted-flow adjoint**, whose exact per-layer formulas are (verified term-for-term against `rsf_stack_invert_row` and `rsf_stack_backward_gradients_fused` in `src/hw/accel/main.fut`; `o` = layer output side, `x` = layer input side, `u` = the pre-`R` coupling output recovered from `o`):

```
u1 = (o1 + o2)/√2,                       u2 = (o2 − o1)/√2
x2 = u2 − W_t·u1 − b_t,                  s = clip(W_s·x2 + b_s),        x1 = u1·exp(−s)
∂L/∂s   = −g1 ⊙ x1                                            (g1 = ∂L/∂x1, zeroed where s saturated)
∂L/∂W_s = ∂L/∂s ⊙ x2,                    ∂L/∂b_s = ∂L/∂s
∂L/∂x2  = g2 + W_s ⊙ ∂L/∂s                                    (total derivative w.r.t. the intermediate x2)
∂L/∂W_t = −∂L/∂x2 ⊙ u1,                  ∂L/∂b_t = −∂L/∂x2
∂L/∂u1  = g1 ⊙ exp(−s) − W_t ⊙ ∂L/∂x2,   ∂L/∂u2 = ∂L/∂x2
g_{o1}  = (∂L/∂u1 − ∂L/∂u2)/√2,          g_{o2} = (∂L/∂u1 + ∂L/∂u2)/√2
```

  where every `∂L/∂s` additionally carries the volume term `−ld_shift` with `ld_shift = logdet_weight / valid_tokens`, matching the existing fused kernel's sign convention, and where `Θ` contributes its own adjoint (itself) at the position prescribed by Section 4.5 when diffusion is enabled.
- Gradient partitioning: layers `0..M−1` receive gradients from the forward frontier's adjoint; layers `M..L−1` receive gradients from the backward frontier's inverted-flow adjoint above. Both write into the same per-layer `[dim, 2]` gradient stacks, so the returned gradient arrays have the identical shape and meaning as `rsf_stack_backward_gradients_fused`.
- Auxiliary memory: the backward frontier needs no stored activations — it reconstructs `x2`, `s`, `x1`, `u1`, `u2` from `o` and the weights at each step, exactly as the existing fused backward does. `O(1)` per row.

### 4.8 Causal cross-token relational coupling

Token isolation is removed by a strictly lower-triangular binary bitmask coupling `C ∈ {0,1}^{T×T}` with `C_{i,j} = 0` for all `j ≥ i` (`types.RSFSequenceMask`):

```
K_t   = Σ_{t' < t} C_{t,t'} · X_{2,t'}                       (prefix accumulation over earlier tokens, per channel)
s_t   = clip(W_s·K_t + b_s, clip_min, clip_max)
Y_1,t = X_1,t · exp(s_t)
Y_2,t = X_2,t + W_t·Y_1,t + b_t
```

- **Exact inverse with no stored trajectory**, all three steps parallel over `t` except the prefix sums:
  1. `X_2,t = Y_2,t − W_t·Y_1,t − b_t` (uses only `Y`, no `K` needed),
  2. re-evaluate `K_t = Σ_{t'<t} C_{t,t'}·X_2,t'` from the recovered `X_2`, then `s_t = clip(W_s·K_t + b_s)`,
  3. `X_1,t = Y_1,t · exp(−s_t)`.
- **Volume**: the Jacobian is block lower-triangular in the token index with diagonal blocks `[[diag(exp(s_t)), 0], [*, I]]`, hence `log|det J| = Σ_t Σ_d s_{t,d}` — analytic, additive, no approximation. The clip does not break bijectivity (the inverse recomputes the same `s_t` from the recovered `X_2`); it only zeroes the gradient w.r.t. `s_t` where saturated, exactly as in the per-row coupling.
- **Causality**: token `t` output depends only on tokens `t' ≤ t`; verified by test (gate 13).
- This is an RSF coupling mode, not a separate mechanism: it reuses the same `W_s`, `W_t` `[dim, 2]` parameter tensors of a layer and the same clip bounds, and it is selected by passing a mask. Without a mask the model behaves exactly as before (per-token rows).
- **Work complexity, stated honestly.** For a *general* strictly lower-triangular mask the prefix accumulation costs `O(nnz·dim)` per layer, where `nnz` is the number of set bits; for the full causal mask `nnz = seq_len·(seq_len−1)/2`, i.e. `O(seq_len²·dim)` — the same order as an attention score matrix, but with no learned projections, no softmax, and an exact closed-form inverse. Only the **full causal mask** admits the `O(seq_len·dim)` running recurrence `K_t = K_{t−1} + X_{2,t−1}`, and it may be used only when `mask.full_causal` is true; the implementation branches on that flag **once per traversal**, never per element. The measured cost must be reported in the Phase 12 benchmark.

### 4.9 Exact rank-2 spectral normalization

Coupling matrices are `M × 2`, so `G = W^T W` is exactly 2×2 and `σ_max` has a closed form; the 30-iteration power method is unnecessary computation and it *underestimates* `σ_max`, which makes the constraint inconsistent between CPU and device:

```
a = Σ_i W_{i,0}²,   b = Σ_i W_{i,0}·W_{i,1},   c = Σ_i W_{i,1}²      (accumulate in f64)
tr = a + c,   diff = a − c,   Δ = diff² + 4b²
λ_max = (tr + √Δ)/2,   σ_max = √max(0, λ_max)
if σ_max > σ_target:  W ← W · (σ_target / σ_max)
```

Scope rule (normative): the closed form is exact **only** for two-column matrices. It replaces the power iteration everywhere the operand is a coupling matrix (`[dim, 2]` per layer, or `[layers][dim][2]` stacks). The dense embedding matrix in `embedding_spectral_normalize` / `applyEmbeddingSpectralNormalization` is `[vocab_size][dim]` with `dim ≫ 2`; for it the 2×2 Gram formula would return a mathematically wrong `σ_max`, so that path **keeps** its existing iterative normalization unchanged. Do not "unify" the two; document the distinction at both call sites.

## 5. Phase 0 — Baseline capture, build-graph correction, and performance/theory baseline

Do this before touching any module code. Every number recorded here becomes an acceptance gate later; record raw command output, never a summary you wrote from memory.

1. Toolchain and baseline — confirm the verified state, do not assume it: `zig version` must print `0.14.1` for `/home/user/venv/bin/zig`. Re-test the Futhark install (`curl -LO https://github.com/diku-dk/futhark/releases/download/v0.26.4/futhark-0.26.4-linux-x86_64.tar.xz`, verify sha256 `bd07eba4c8f2b39ed7b494bf880a3fc5fe46254cd0cabd8ca1a63e5cf240f300`, `tar xf`, `cd futhark-0.26.4 && PREFIX=/home/user/.local make install`, then `zig build regen-futhark` to create the gitignored `src/hw/accel/{main_cpu,main_gpu,futhark_kernels}.{c,h}`); in this sandbox it fails with `SSL_ERROR_SYSCALL` from the release CDN. Record verbatim into `docs/rsf_native_baseline.txt`: (i) `zig build check`, `zig build c-api`, `zig build test-c-api`; (ii) the direct spec sweep `for w in src/test_root_*.zig; do zig test "$w"; done`, whose verified result today is 18 of 19 green with 562 tests total (per-spec counts in Section 2) and `test_root_rsf.zig` failing with `no module named 'build_options'`, or, once an options module is supplied, with undefined `futhark_*` symbols at link time; (iii) `zig build test-all` and `zig build -Doptimize=ReleaseFast bench`, both stopping at `run futhark`; (iv) `futhark --version` output or its absence. Every blocked command is recorded with its exact failing step name and error text. Baseline numbers used by the Phase 14 gate comparison come only from runs that actually executed; blocked gates are listed as blocked, never as passed and never as estimated.
2. Record the **iterative-cost baseline** that Phases 1–6 must beat, into `docs/upgrade/baseline-bench.txt`:
   - (a) Wall clock of `spectralNormPowerIteration` + `constrainSpectralNorm` for one `[49152, 2]` coupling stack at 30 iterations (ReleaseFast, 100 repetitions, median), and of `accel.RSFAccelerator.spectralNormalizeLayers(target, 30)` when a device is available. **Verified blocker:** both functions live in `src/processor/rsf.zig`, whose import closure reaches `src/hw/accel/accel_interface.zig`, so no harness that imports `rsf.zig` can link without the generated Futhark C. If Futhark is unavailable, take the baseline through a standalone harness rooted at `src/` (e.g. `src/tests/harness_spectral_baseline.zig`, registered as a `bench`-style executable step *without* `applyAccel`) that imports only `src/core/tensor.zig` and runs the identical 30-iteration power method on an identically generated `[49152, 2]` stack — record in `docs/upgrade/baseline-bench.txt` that this measures the same algorithm outside `rsf.zig`, name the harness file, and state plainly that the in-place `rsf.zig` measurement was blocked. Do not present the substitute as the blocked measurement.
   - (b) Wall clock of one fused single-frontier training step (`trainPreparedStepFuthark` on the CPU path, or `accel.fusedTrainingStep` on device) at `L = 24`, `dim = 512`, batch 4, seq 64 — the reference for gate 9.
   - (c) **SFD diagonal convergence baseline**: with the current `sfd.zig`, a deterministic scenario (dim 8, 2 layers, fixed PRNG seed 0x5EED, fixed target batch of 64 rows, lr from `SFDConfig` defaults) — record the number of steps needed to reach 50% of the initial mean squared error, and the loss curve at steps 1, 10, 50, 100, 200, into `docs/upgrade/sfd-convergence.txt`. This exact scenario is re-run in Phase 4 and Phase 13 for gate 11; write it as a small standalone harness under `src/tests/harness_sfd_convergence.zig` (registered as a `bench`-style executable step `sfd-convergence` in `build.zig`) so it is reproducible before and after the rewrite. **The harness's API usage changes in Phase 4** (the current `SFD.init(allocator, param_size)` disappears), so what must survive the rewrite is the *scenario definition*, recorded verbatim in `docs/upgrade/sfd-convergence.txt`: model `dim`/`num_layers`, PRNG seed, how the target batch is generated, `lr`, every `SFDConfig`/`TrainerConfig` value used, the target-loss threshold, and the step cap. Capture the baseline in Phase 0 with the harness written against the current API; rewrite the harness against the new API in Phase 4; re-run the identical scenario. If a scenario parameter cannot be expressed in the new API, record that fact and the substitute used — never silently change the scenario to improve the ratio.
   - (d) Wall clock of `bitmaskSignalPropagate` on a fixed 4096-node adjacency bitmask with a fixed PRNG signal (the reference for Phase 11).
   - (e) The diffusion layout of every dimension used by tests and by production defaults: for `model_dim = 98304` → `row_len = 98304, r = 3, block = 32768, stages = 15`; record the same triple for each test dim used later (16, 32, 512, …) by computing `rsfDiffusionLayout(2·dim)`.
3. Produce golden fixtures with the current code and store them under `docs/upgrade/fixtures/`:
   - (a) an RSF model saved via `RSF.save` (dim 16, 3 layers, `SAVE_VERSION 6`) plus its forward/inverse/log-det outputs on a fixed PRNG batch;
   - (b) an SSI `serialize` dump for a fixed token corpus;
   - (c) a Ranker `exportModel` dump;
   - (d) an SFD `saveState` dump (magic `0x53464433`);
   - (e) a trainer-checkpoint fixture (`CHECKPOINT_VERSION = 7`) built from `accel.RSFOptimizerState` populated with deterministic arrays including `fisher_s`/`fisher_t` (no GPU required — construct and serialize the state object directly);
   - (f) an OFTB golden: `forwardSliceInPlace` and `backwardSliceInPlace` outputs for a fixed 2·16 and a fixed 2·49152-length vector (the pre-diffusion reference that `global_diffusion = false` must reproduce exactly);
   - (g) an nsir golden: `bitmaskSignalPropagate` outputs for a fixed 256-node bitmask and fixed signal (the exact-scalar reference for Phase 11);
   - (h) a causal-coupling golden: none exists pre-upgrade (the mode is new) — instead record the per-token forward outputs of the *non-causal* path on the same weights, which the causal path must reproduce when the mask is all-zero.
   These fixtures are the compatibility targets for the per-phase legacy-loader tests (Phases 3–12) and the final golden-compatibility test (Phase 13).
4. Build-graph correction in `build.zig`:
   - Remove the dead registrations: the `core_relational_mod`, `tensor_core_mod`, and `tokenizer_mod` module creations and every `addImport("core_relational", …)`, `addImport("tensor_core_matmul", …)`, `addImport("tokenizer", …)` call. (Verified unused; removal eliminates duplicate-module hazards when `vpu.zig`/`r_gpu.zig` gain `../../core/tensor.zig` imports.)
   - Add one new module: `const jaide_mod = b.createModule(.{ .root_source_file = b.path("src/lib_root.zig"), .target, .optimize }); jaide_mod.addOptions("build_options", build_options);` with a new file `src/lib_root.zig` re-exporting the RSF family: `pub const types = @import("core/types.zig"); pub const tensor = @import("core/tensor.zig"); pub const oftb = @import("processor/oftb.zig"); pub const rsf = @import("processor/rsf.zig"); pub const sfd = @import("optimizer/sfd.zig"); pub const ssi = @import("index/ssi.zig"); pub const ranker = @import("ranker/ranker.zig"); pub const nsir_core = @import("core_relational/nsir_core.zig");` (rooted directly in `src/`, so all relative imports inside the tree are legal).
   - Register `jaide_mod` as the import named `jaide` on: the RTL executable (`rtl_exe.root_module.addImport("jaide", jaide_mod)`), the bench module (`bench_deps.addImport("jaide", jaide_mod)`), the new `sfd-convergence` harness, and every test artifact and executable produced by `applyAccel` (extend `AccelConfig` with the module and add it in `applyAccel`). Update `src/hw/rtl/rtl_sim_main.zig` to import SSI as `@import("jaide").ssi.SSI` instead of the escaping `../../index/ssi.zig`.
   - Decouple the specs from Futhark codegen according to their verified import closures: of the 19 wrappers only `src/test_root_rsf.zig` reaches `src/hw/accel/accel_interface.zig` (via `src/processor/rsf.zig`), and `src/test_root_gpu_memory.zig` reaches `hw/accel/gpu_memory.zig`/`compact_batch.zig` without touching any Futhark extern (it links and passes today with plain `zig test`). Replace the blanket `applyAccel(test_artifact, accel, gpu_enabled)` at `build.zig:258` with an explicit per-spec policy: apply `applyAccel` only to `test-rsf`, and give the other 18 artifacts just what they actually need — `artifact.root_module.addOptions("build_options", build_options)` where their closure reads it, plus the `core_relational` / `tensor_core_matmul` imports if used — with no `addCSourceFile` and no `dependOn(codegen)`. Verified target state: `zig build test-all` runs 18 specs green without the `futhark` binary and reports only `test-rsf` as blocked, instead of blocking all 19 at `run futhark`. Leave `applyAccel`'s own body (Futhark/CUDA wiring, `-DJAIDE_FUTHARK_CUDA`, generated C sources) unchanged; change only which artifacts it is applied to. Every artifact registered later in this upgrade follows the same policy: `sfd-convergence`, the spectral baseline harness, and `test-rgpu` are registered without `applyAccel` because their closures stay accelerator-free; `test-rsf-native` (Phase 13) is split into an accel-free spec holding every invariant that does not need the accelerator and an accel-backed spec for the checks that import `src/processor/rsf.zig`, so that the invariant suite stays runnable in this environment.
   - Keep `applyAccel`'s Futhark/CUDA wiring exactly as is (generated C sources, `-DJAIDE_FUTHARK_CUDA`, `build_options`), and ensure the Futhark compile step is re-run whenever `src/hw/accel/*.fut` changes in Phases 5–6 (if the build caches generated C, regenerate explicitly and record the Futhark compiler version used).
5. Checkpoint (the commit-equivalent state before proceeding): `zig build check` passes, `zig build c-api` and `zig build test-c-api` pass, and after the build-graph correction `zig build test-all` runs the 18 accel-free specs green with `test-rsf` reported as blocked-on-Futhark (or green too, if Futhark was installed). Record all three outputs plus the direct `zig test src/test_root_*.zig` sweep into `docs/rsf_native_baseline.txt`. A regression in any of the 562 currently-passing tests stops the upgrade here.

## 6. Phase 1 — `src/core/tensor.zig`: the RSF-native tensor engine

Rewrite the file as the system's RSF tensor engine while preserving every existing public symbol used elsewhere (check with grep before removing anything; `GemmReport`, `HugePageAllocator`, `packA`/`packB`, `benchmarkGemm`, `MatmulComptime`, `TensorIterator`, all `Tensor` methods listed in Section 2 remain).

Additions (exact contract):

1. `Tensor` gains a field `rsf: ?types.RSFBinding = null` plus constructors and validators:
   - `pub fn initBound(allocator: Allocator, dims: []const usize, binding: types.RSFBinding) !Tensor`
   - `pub fn initCoupling(allocator: Allocator, binding: types.RSFBinding) !Tensor` — shape `[binding.dim, 2]`, zero-filled (for `.layer_weight_s`, `.layer_weight_t`, `.gradient`, `.fisher`, `.momentum`, `.master_weight`, `.ranker_head`).
   - `pub fn initFisherBlocks(allocator: Allocator, binding: types.RSFBinding) !Tensor` — shape `[binding.dim, 3]`, space `.fisher_block`, zero-filled.
   - `pub fn initLatent(allocator: Allocator, model_id: u64, dim: usize, batch: usize) !Tensor` — shape `[batch, dim, 2]`, `.latent_state`.
   - `pub fn initIndexPayload(allocator: Allocator, model_id: u64, dim: usize) !Tensor` — length `2*dim`, `.index_payload`.
   - `pub fn binding(self: *const Tensor) ?types.RSFBinding`
   - `pub fn requireSpace(self: *const Tensor, space: types.RSFSpace) types.RSFBindingError!types.RSFBinding` — errors `RSFBindingRequired` when null, `RSFSpaceMismatch` when the space differs.
   - `pub fn requireCouplingCompatible(self: *const Tensor, other: *const Tensor) types.RSFBindingError!void` — same `model_id`, same `dim`, spaces compatible for the operation.
   - `clone`/`copyFrom` must propagate the binding; COW semantics unchanged.
2. Canonical coupling kernels (vectorized, `@Vector(8, f32)` with the existing AVX-512 16-lane detection pattern used by `OFTB.vectorLen()`, scalar tail loop; all functions take explicit `clip_min`/`clip_max` and parameter slices strided from `[dim, 2]` tensors):
   - `pub fn couplingScaleRow(s_w: []const f32, s_b: []const f32, x2: []const f32, clip_min: f32, clip_max: f32, out_scale: []f32) void` and `pub fn couplingScaleLogDetRow(...) f32` (returns `sum_d c[d]`).
   - `pub fn couplingTranslationRow(t_w: []const f32, t_b: []const f32, x1: []const f32, out_trans: []f32) void`.
   - `pub fn couplingForwardRow(row: []f32, s_params, t_params, clip_min, clip_max, scale_buf: []f32, trans_buf: []f32) !void` — in-place on a `2*dim` row, exactly the Section 4.2 algebra.
   - `pub fn couplingForwardLogDetRow(...) !f32`.
   - `pub fn couplingInverseRow(row: []f32, ...) !void`.
   - `pub fn couplingAdjointRow(x1_row, x2_row, dy1_row, dy2_row, dx1_out, dx2_out, s_w, s_b, t_w, t_b, clip_min, clip_max, ds_buf, y1_buf, dy1_total_buf, grad_scale: f32, logdet_adjoint: f32, s_grad: ?[]f32, t_grad: ?[]f32) !void` — the full adjoint including parameter-gradient accumulation, mirroring `LayerCore.backwardFromInputsRow` semantics bit-for-bit in structure (saturation zeroes `ds`'s data term but keeps `logdet_adjoint`).
   - `pub fn couplingInvertedFlowAdjointRow(o1_row, o2_row, g1_row, g2_row, s_w, s_b, t_w, t_b, clip_min, clip_max, grad_scale: f32, ld_shift: f32, go1_out, go2_out, s_grad: ?[]f32, t_grad: ?[]f32, scratch: *InvertedFlowScratch) !void` — the inverted-flow adjoint of Section 4.7 term-for-term (`u1`, `u2`, `x2`, `s`, `x1`, `∂L/∂s`, `∂L/∂x2`, `∂L/∂u1`, `∂L/∂u2`, `g_{o1}`, `g_{o2}`, weight gradients), where `InvertedFlowScratch` is a reusable `dim`-sized buffer struct (`init(allocator, dim)`, `deinit`) so a whole frontier traversal allocates once.
   - Batch forms operating on `[batch, dim, 2]` tensors: `pub fn couplingForwardBatch(state: *Tensor, s_weight: *const Tensor, t_weight: *const Tensor, clip_min: f32, clip_max: f32) !f32` (returns summed log-det), `couplingInverseBatch`, `couplingAdjointBatch(state, inputs, s_weight, t_weight, s_grad, t_grad, grad_scale, logdet_adjoint) !void`, `couplingInvertedFlowAdjointBatch(...)` — all with binding validation (`.latent_state` state, `.layer_weight_*`/`.gradient` params, matching `model_id`/`dim`).
3. **Exact rank-2 spectral normalization** (replaces every power iteration on coupling matrices; Section 4.9):
   - `pub fn exactSpectralNormRank2(c0: []const f32, c1: []const f32) !f32` — f64 accumulators for `a`, `b`, `c`; `error.InvalidDimension` on empty or mismatched lengths.
   - `pub fn couplingSpectralNorm(weight: *const Tensor) !f32` — validates shape `[rows, 2]` (`.layer_weight_s`, `.layer_weight_t`, `.gradient`, `.master_weight`, `.ranker_head` accepted) and reads the two columns with the tensor's row stride.
   - `pub fn constrainCouplingSpectralNorm(weight: *Tensor, target: f32) !f32` — in-place rescale when `σ_max > target`, returns `σ_max` before scaling; no allocator, no seed, no iteration count, no PRNG. The old `iterations`/`seed` parameters are removed from this API everywhere.
   - `pub fn constrainCouplingStackSpectralNorm(weights: []Tensor, target: f32) !usize` — per-tensor application over a layer stack, returns the number of tensors actually rescaled.
4. **Global diffusion kernels** (Section 4.5):
   - `pub fn walshHadamardInPlace(data: []f32) !void` — normalized FWHT, `error.InvalidDimension` unless `len` is a power of two and non-zero; butterfly stages `h = 1, 2, …, len/2` with `(x_j + x_{j+h})/√2`, `(x_j − x_{j+h})/√2`; vectorized over `j` with `@Vector` when `h ≥ VLEN` (the vector lanes are the inner pair loop, never the stage loop), scalar otherwise.
   - `pub fn mixRadixBlocksInPlace(data: []f32, layout: types.RSFDiffusionLayout) !void` — the `Q_r` factor: per offset `o`, compute `S[o]` by summing the `r` block values, then write `x[b·m+o] − (2/r)·S[o]`; when `r == 1` this reduces to `x ← −x`, which is still an exact involution with `|det| = 1` and must be implemented (not skipped).
   - `pub fn diffuseRowInPlace(row: []f32, layout: types.RSFDiffusionLayout) !void` — `Θ = Q_r ⊗ H_{2^k}`: validate `row.len == layout.row_len`, apply `walshHadamardInPlace` to each of the `r` contiguous blocks, then `mixRadixBlocksInPlace`. Contract required by gate 4(b): operate strictly in place on the caller's buffer, touch the row at most twice per call (the butterfly stages in place, then the `Q_r` pass), and allocate nothing per stage.
   - `pub fn diffuseBatchInPlace(state: *Tensor, layout: types.RSFDiffusionLayout) !void` — applies `Θ` to every row of a `.latent_state` tensor.
   - `pub fn diffusionLayoutFor(dim: usize) !types.RSFDiffusionLayout` — wraps `types.rsfDiffusionLayout(2*dim)` and returns `error.InvalidDimension` for `dim == 0`.
5. **Causal bitmask coupling kernels** (Section 4.8):
   - `pub fn causalPrefixSum(mask: *const types.RSFSequenceMask, x2: []const f32, dim: usize, out_K: []f32) !void` — `out_K[t·dim+d] = Σ_{t'} C[t,t']·x2[t'·dim+d]`, with two implementations selected **once per call** from `mask.full_causal`: (i) full causal mask → the running recurrence `K_0 = 0`, `K_t = K_{t−1} + X_{2,t−1}`, cost `O(seq_len·dim)`; (ii) general mask → iterate each row's set bits with `mask.rowSetBits` and accumulate, cost `O(nnz·dim)`. A running accumulator is **not** valid for a general mask (different rows have different support); taking path (i) for a non-full mask is a correctness defect and must be caught by a test that runs both paths on a full causal mask and compares them.
   - `pub fn causalCouplingForwardBatch(state: *Tensor, mask: *const types.RSFSequenceMask, s_weight: *const Tensor, t_weight: *const Tensor, clip_min: f32, clip_max: f32, scratch_K: []f32) !f32` — per token: `K_t` from `causalPrefixSum`, `s_t = clip(W_s·K_t + b_s)`, `Y_1,t = X_1,t·exp(s_t)`, `Y_2,t = X_2,t + W_t·Y_1,t + b_t`; returns `Σ_t Σ_d s_{t,d}`. `scratch_K` is caller-owned, `seq_len·dim` long, and reused across layers.
   - `pub fn causalCouplingInverseBatch(state: *Tensor, mask, s_weight, t_weight, clip_min, clip_max, scratch_K: []f32, scratch_X2: []f32) !void` — the three-step exact inverse of Section 4.8; `scratch_X2` holds the recovered `X_2` for the whole sequence (needed to re-evaluate `K`), and no other auxiliary storage is used: two buffers total, both independent of `num_layers`.
   - `pub fn causalCouplingAdjointBatch(state_in, state_out, grad_out, mask, s_weight, t_weight, s_grad, t_grad, clip_min, clip_max, grad_scale, logdet_adjoint, scratch) !void` — the adjoint of the causal map: `∂L/∂s_t` (zeroed where saturated, plus `logdet_adjoint`), `∂L/∂X_1,t = ∂L/∂Y_1,t·exp(s_t)`, and the **transpose accumulation** `∂L/∂X_2,t' += Σ_{t>t'} C[t,t']·W_s[d]·∂L/∂s_{t,d}`, implemented as the descending recurrence `G_{t'} = G_{t'+1} + W_s ⊙ ∂L/∂s_{t'+1}` when `full_causal` (`O(seq_len·dim)`) and by iterating set bits otherwise (`O(nnz·dim)`), plus `∂W_s += ∂L/∂s_t ⊙ K_t`, `∂b_s += ∂L/∂s_t`, `∂W_t += ∂L/∂X_2,t ⊙ Y_1,t`, `∂b_t += ∂L/∂X_2,t`. Each kernel's total cost is the accumulation cost above plus the coupling's own `O(seq_len·dim)`.
   - All of the above validate bindings and mask dimensions (`error.RSFDimMismatch`, `error.DimensionMismatch`, `error.InvalidCausalMask`).
6. Even/odd accessors on latent tensors: `pub fn evenHalf(self: *Tensor) ![]f32`, `pub fn oddHalf(self: *Tensor) ![]f32`, `pub fn rowHalves(self: *Tensor, row: usize) !struct { x1: []f32, x2: []f32 }`.
7. GPU-path helpers: `pub fn toFP16(src: *const Tensor, dst: []f16) !void` / `pub fn fromFP16(src: []const f16, dst: *Tensor) !void` (move from `sfd.zig`; keep the clamp to ±65504).
8. Tests (colocated, run under `test-tensor`):
   - binding validation errors;
   - coupling kernel equivalence against a scalar reference implementation of the Section 4.2 formulas over a PRNG sweep (≥ 10,000 rows, dims 1..257 odd/even, tolerance abs 1e-5 / rel 1e-4);
   - forward∘inverse roundtrip ≤ 1e-4 relative; adjoint correctness against finite differences on dim ≤ 8 (parameter and input gradients, tolerance rel 1e-3);
   - **inverted-flow adjoint** (`couplingInvertedFlowAdjointRow`) against finite differences of the inverse map on dim ≤ 8, tolerance rel 1e-3, including the `ld_shift` volume term and the saturation zeroing;
   - **exact spectral norm**: for 1,000 PRNG coupling matrices of shapes `[2,2]`, `[17,2]`, `[49152,2]`, `exactSpectralNormRank2` matches two *independent* references within `2.0e-6` absolute — (i) an f64 Jacobi eigenvalue iteration on `G = WᵀW`, and (ii) a brute-force angular sweep `max_θ ‖W(cosθ, sinθ)ᵀ‖` over 2·10⁶ samples; `constrainCouplingSpectralNorm` lands at `σ_max ≤ target·(1+1e-6)` and leaves matrices already below target bit-identical;
   - **diffusion**: `Θ(Θ(x)) = x` within `1e-7` (∞-norm) for `row_len ∈ {2, 4, 8, 24, 32, 96, 32768·3}`; `|‖Θ(x)‖₂ − ‖x‖₂| < 1e-7`; `Θ` applied to a unit basis vector produces non-zero output in every one of the `r` blocks (cross-channel mixing); `r == 1` case (pure power-of-two row) still involutive; `walshHadamardInPlace` rejects non-power-of-two lengths with `error.InvalidDimension`;
   - **causal coupling**: forward∘inverse roundtrip `max(‖X1−X1_rec‖_∞, ‖X2−X2_rec‖_∞) < 5.0e-7` on seq ≤ 64, dim ≤ 32; all-zero mask reproduces the per-token non-causal coupling exactly (Phase 0 fixture (h)); log-det equals `Σ_t Σ_d s_{t,d}`; causality (mutating token `t'` leaves outputs at `t < t'` bit-identical); adjoint against finite differences on seq ≤ 5, dim ≤ 4, tolerance rel 1e-3; a non-lower-triangular mask is rejected with `error.InvalidCausalMask`; plus path equivalence: on a full causal mask, the running-recurrence path and the set-bit-iteration path of `causalPrefixSum` produce identical `K` within `1e-6`, and a relational (non-full) mask is provably routed through the set-bit path (assert via an instrumented counter or by a mask whose rows have different support, where a running accumulator would give a wrong answer).
   - huge-page and COW paths still pass the existing tests; new benchmark cases in `bench_tensor_ops.zig` for `couplingForwardBatch`/`couplingInverseBatch`/`diffuseRowInPlace`/`causalCouplingForwardBatch` (added in Phase 12).

## 7. Phase 2 — `src/processor/oftb.zig`: global cross-channel diffusion

The OFTB becomes the orthogonal factor `R` **plus** the global diffusion `Θ` of Section 4.5, and nothing else. `FRACTAL_SCALE`, `LOG_DET_JACOBIAN = 0.0`, `vectorLen()`, and the existing rotation formulas stay exactly as they are.

1. `OFTB` gains two fields: `diffusion_enabled: bool` and `layout: types.RSFDiffusionLayout` (computed from `2·dim` at init).
   - `pub fn init(d: usize) OFTB` keeps its current signature and semantics and sets `diffusion_enabled = false` — this preserves the Phase 0 fixture (f) and every existing `test-oftb` expectation for legacy models.
   - `pub fn initDiffusing(d: usize) !OFTB` and `pub fn initWithDiffusion(d: usize, enabled: bool) !OFTB` set the flag explicitly and precompute the layout; they return `error.InvalidDimension` for `d == 0` and `error.DimensionOverflow` when `2·d` overflows.
2. `pub fn fastWalshHadamardTransformInPlace(data: []f32) error{InvalidDimension}!void` — the `H_m` primitive, as a public **delegating wrapper** around `tensor.walshHadamardInPlace` (Phase 1): the butterfly arithmetic lives in exactly one place (`src/core/tensor.zig`, vectorized), and `OFTB` exposes the name and contract the source plan requires. Do not maintain a second butterfly implementation.
   - Validate `data.len > 0` and `(len & (len − 1)) == 0`, returning `error.InvalidDimension` otherwise. **Deviation from the source plan, required and deliberate**: the plan's sketch uses `std.debug.assert`; a public library function must return a typed error instead of aborting the process on invalid input (Section 3, constraint 6). The arithmetic is otherwise identical to the sketch: `inv_sqrt2 = 0.7071067811865476`, stages `h = 1, 2, 4, …` while `h < n`, inner pair loop over `i` in steps of `2h` and `j` from `i` to `i + h`, writing `(u + v)·inv_sqrt2` and `(u − v)·inv_sqrt2`.
   - The tensor kernel vectorizes the `j` loop with `@Vector(vectorLen(), f32)` when `h ≥ vectorLen()` and finishes with the scalar tail; stage order is ascending `h`, identical to the device kernel of Phase 5 so that CPU and GPU results agree.
3. `pub fn mixRadixBlocksInPlace(self: OFTB, data: []f32) !void` — delegates to `tensor.mixRadixBlocksInPlace(data, self.layout)` (the `Q_r` factor).
4. `pub fn diffuseSliceInPlace(self: OFTB, data: []f32) !void` — `Θ` on one `2·dim` slice; a no-op when `diffusion_enabled == false`; validates `data.len == 2·dim`.
5. Integration, with the exact order mandated by Section 4.5:
   - `forwardSliceInPlace`: rotation `R` first (existing code, unchanged), then `diffuseSliceInPlace`. Signature becomes `!void` (it can now fail on dimension validation); update `forwardInPlace` and every caller.
   - `backwardSliceInPlace`: `diffuseSliceInPlace` first, then the existing adjoint rotation `R^T`. Signature becomes `!void`. Because `Θ^T = Θ` and `R^T = R⁻¹`, this one function remains simultaneously the inverse map (`inverseSliceInPlace`, `inverseInPlace`) and the gradient adjoint — state this identity in the doc comment and cover it with a test.
   - `forwardBackwardFusedInPlace` and `symplecticReversalInPlace` are updated to the same ordering (activation: `R` then `Θ`; gradient: `Θ` then `R^T`) and to the error-returning signatures.
6. Structural facts to expose and test (Section 4.4):
   - `pub const ROTATION_ORDER: usize = 8;` plus `pub fn rotationIsIdentityAfterOrder(self: OFTB, data: []f32) !bool` — applies `R` eight times (with diffusion disabled internally for this check) and compares to the input within `1e-6`.
   - `pub fn diffusionIsInvolution(self: OFTB, data: []f32, allocator: Allocator) !bool` — `Θ(Θ(x)) = x` within `1e-7`.
   - `pub fn logDetContribution(self: OFTB) f32 { return 0.0; }` — documented as exact: `det R = 1`, `|det Θ| = 1`.
7. Tests (under `test-oftb`, existing tests preserved): legacy `init` reproduces fixture (f) bit-for-bit; `initDiffusing` on `dim = 16384·3/…` (choose `dim = 16` → `row_len = 32`, `r = 1`, `stages = 5`; and `dim = 12` → `row_len = 24`, `r = 3`, `block = 8`, `stages = 3`; and `dim = 49152` → `row_len = 98304`, `r = 3`, `block = 32768`, `stages = 15`) verifies the layouts; involution and energy preservation of the combined `Θ∘R` map to `1e-6`; `forwardSliceInPlace` then `backwardSliceInPlace` recovers the input within `1e-6` for both flag states; `R^8 = I` within `1e-6`; non-power-of-two and zero lengths return typed errors, never panic.

## 8. Phase 3 — `src/processor/rsf.zig`: substrate exposure, exact spectral norm, midpoint frontiers, causal flow

Modify (do not redesign) `rsf.zig`:

1. Replace the private scalar implementations `computeScaleRow`, `computeTranslationRow`, `computeScaleLogDetRow`, `couplingForwardRow`, `couplingForwardLogDetRow`, `couplingInverseRow`, and the row loop inside `backwardFromInputsRow` with calls to the `tensor.zig` kernels. Keep the public behavior, error names, and tolerances identical (the GPU cross-check `5e-2` and `verifyInvertible` defaults must still pass). Update every OFTB call site to the error-returning signatures from Phase 2, and construct the core's `OFTB` with `initWithDiffusion(dim, cfg.global_diffusion)`.
2. **Exact spectral normalization** (Section 4.9):
   - Add `pub fn exactSpectralNormRank2(c0: []const f32, c1: []const f32) !f32` as a thin re-export/delegation to `tensor.exactSpectralNormRank2` (one implementation, two visible names for API continuity — the delegation must be a call, not a copy).
   - Delete `spectralNormPowerIteration` and the constant `LAYER_SPECTRAL_POWER_ITERATIONS`; keep `LAYER_TARGET_SPECTRAL_NORM = 0.9`. `LayerCore.constrainSpectralNorm` calls `tensor.constrainCouplingSpectralNorm(weight, LAYER_TARGET_SPECTRAL_NORM)` — no allocator, no seed, no iteration count. Any public API that took an `iterations` argument for coupling normalization loses that argument; grep every caller (including `sfd.SpectralNormalizer` in Phase 4 and `accel` in Phase 6) and update it.
   - Record in the doc comment why the closed form is used: single pass, exact, no numerical underestimation, identical result on CPU and device.
3. **Configuration and save format**:
   - `RSFConfig` gains exactly one new field, `global_diffusion: bool = true`. `RSFLayerConfig` is unchanged. The causal coupling mode of item 7 is selected per call by passing a mask, not by configuration, so no causal config field is added to `RSFConfig` (the trainer's `TrainerConfig.causal_sequence_coupling` in Phase 12 is the only causal switch, and it governs the training loop, not the model). Validate `global_diffusion` only against dimension feasibility via `tensor.diffusionLayoutFor(dim)` (always feasible for `dim ≥ 1`; return `error.InvalidDimension` for `dim == 0`).
   - Bump `SAVE_VERSION` to 7. The v7 snapshot adds: (i) the `global_diffusion` flag, (ii) the per-layer Gauss–Newton block accumulators (optional section; default zero), (iii) nothing else — all v6 fields stay byte-identical. `loadWithConfig` reads v6 (diffusion flag absent → `false`; new section absent → zeros) and v7. Add a test that loads the Phase 0 golden v6 model (fixture (a)) and reproduces its recorded forward, inverse, and log-det outputs within the golden tolerances, and a test that the same weights with `global_diffusion = true` produce *different* outputs (proving the flag is real and load-bearing).
4. All layer tensors (`s_weight`, `t_weight`, gradients) are created with bindings (`.layer_weight_s`/`.layer_weight_t`/`.gradient`, model id, layer index, dim); Fisher block tensors created by SFD use `.fisher_block`. Model input/output tensors passed to `forward`/`inverse`/`backward` get `.latent_state` bindings attached by new internal helpers; the existing public signatures stay, and new bound-typed entry points are added:
   - `pub fn dim(self: *const RSF) !usize`, `pub fn layerCount(self: *const RSF) !usize`, `pub fn diffusionLayout(self: *const RSF) !types.RSFDiffusionLayout`.
   - `pub fn forwardLatent(self: *RSF, state: *RSFLatentState) !void`, `pub fn inverseLatent(...)`, `pub fn forwardLatentWithLogDet(self: *RSF, state: *RSFLatentState) !f32`, `pub fn backwardLatent(self: *RSF, grad_output: *const RSFLatentState, input: *const RSFLatentState, output: *const RSFLatentState, grad_input_out: *RSFLatentState, logdet_weight: f32) !void` — thin bound wrappers over the existing core paths.
5. **Optimizer-facing surface** required by Phase 4 (SFD) and Phase 10 (R-GPU), all registry- and lock-correct:
   - `pub fn readLayerWeights(self: *const RSF, layer: usize, s_out: []f32, t_out: []f32) !void`
   - `pub fn writeLayerWeights(self: *RSF, layer: usize, s_in: []const f32, t_in: []const f32) !void` (exclusive lock; validates finiteness and shape; triggers `refreshGPUAfterWeightChange`)
   - `pub fn readLayerGradients(self: *const RSF, layer: usize, s_out: []f32, t_out: []f32) !void`
   - `pub const RSFBranch = enum { s, t };` and `pub fn readLayerGradientProducts(self: *const RSF, layer: usize, branch: RSFBranch, out: []f32) !void` — returns the raw pair products the block Fisher needs, so no caller recomputes them: `out[d*3+0] = g_w[d]²`, `out[d*3+1] = g_w[d]·g_b[d]`, `out[d*3+2] = g_b[d]²` where `(g_w[d], g_b[d])` are the two columns of the requested branch's `[dim, 2]` gradient tensor. Requires `out.len == dim*3` (`error.InvalidDataLength` otherwise).
   - `pub fn readLayerGaussNewton(self: *const RSF, layer: usize, s_block_out: []f32) !void` — writes the per-layer `[dim, 3]` Gauss–Newton block statistics of the log-det objective for the **s branch only** (`s_block_out.len == dim*3`, `error.InvalidDataLength` otherwise), newly recorded by `LayerCore` during `backwardFromInputsRow` and `backwardWithLogDet`. Because `logdet = Σ_d (s_w[d]·x2[d] + s_b[d])`, the log-det residual for row `d` is linear in `(s_w[d], s_b[d])` with design vector `(x2[d], 1)`, so its Gauss–Newton block is exactly `E_b[(x2[d], 1)ᵀ(x2[d], 1)]`: column 0 = `mean_b(x2[d]²)`, column 1 = `mean_b(x2[d])`, column 2 = `1.0`. There is no t-branch output parameter: the t branch has identically zero log-det curvature, so SFD's `.rsf_gauss_newton` mode must use the plain gradient-product block for it (Phase 4, step 3) — document that fact here and there. Maintain these accumulators in an `?Tensor` of space `.fisher` created by `ensureGradients`, cleared by `zeroGradients`, and averaged over the batch count recorded alongside them.
   - `pub fn scaleLayerGradients(self: *RSF, scale: f32) !void` — multiply every layer's s/t gradients in place under exclusive lock (used by SFD global-norm clipping).
   - `pub fn accumulateLayerGradients(self: *RSF, layer: usize, s_in: []const f32, t_in: []const f32) !void` — element-wise add into the layer's gradient tensors under exclusive lock (used by R-GPU sharded backward and by the midpoint path; validates shapes; calls `ensureGradients` first).
6. Add `RSFLatentState` exactly as specified in Section 4.3, including `roundtripError`.
7. **Causal cross-token sequence flow** (Section 4.8), reusing the layer weights — no new parameters:
   - `pub fn forwardSequence(self: *RSF, state: *RSFLatentState, mask: *const types.RSFSequenceMask) !void` and `pub fn forwardSequenceWithLogDet(...) !f32`, `pub fn inverseSequence(self: *RSF, state: *RSFLatentState, mask: *const types.RSFSequenceMask) !void`, `pub fn backwardSequence(self: *RSF, grad_output: *const RSFLatentState, input: *const RSFLatentState, output: *const RSFLatentState, mask: *const types.RSFSequenceMask, grad_input_out: *RSFLatentState, logdet_weight: f32) !void`.
   - Sequence layout (normative, validated on every call): `state.data` has shape `[tokens, dim, 2]` where `tokens = num_sequences · mask.seq_len`, tokens ordered sequence-major then position-ascending (`token_index = sequence_index·mask.seq_len + t`). Require `tokens % mask.seq_len == 0` and `tokens > 0`, returning `error.DimensionMismatch` otherwise; a single sequence is the `num_sequences == 1` case. Document this layout in the doc comment of every sequence entry point.
   - Auxiliary memory: one `seq_len·dim` `K` buffer plus one `seq_len·dim` recovered-`X_2` buffer per traversal, allocated once per call from a per-core scratch allocator (the existing `scratchAllocator()` pattern), reused across layers — `O(1)` in the number of layers, never `O(seq_len²)`.
   - When `mask` is all-zero the result is bit-identical to the per-token `forward`/`inverse` (test gate, Phase 0 fixture (h)).
8. **Dual-frontier midpoint collision** (Section 4.7):
   - `pub const MidpointResult = struct { z: RSFLatentState, w: RSFLatentState, collision_loss: f32, logdet_forward: f32, logdet_backward: f32, logdet_total: f32, forward_layers: usize, backward_layers: usize };`
   - `pub fn midpointSplit(self: *const RSF) struct { forward_layers: usize, backward_layers: usize }` — `M = L / 2`, `backward = L − M`; for `L == 1` the forward frontier is empty and the backward frontier holds the single layer (document this degenerate case).
   - `pub fn midpointForwardFrontier(self: *RSF, input: *const RSFLatentState, out_z: *RSFLatentState) !void` — applies layers `0..M−1` in order, accumulating `logdet_forward`.
   - `pub fn midpointBackwardFrontier(self: *RSF, target: *const RSFLatentState, out_w: *RSFLatentState) !void` — applies the inverse of layers `M..L−1` in reverse order, accumulating `logdet_backward` as `+Σ clip` (the log-det of each layer, not of its inverse; document the sign convention of Section 4.7).
   - `pub fn midpointCollision(self: *RSF, input: *const RSFLatentState, target: *const RSFLatentState, allocator: Allocator) !MidpointResult` — runs the two frontiers sequentially on the calling thread (identical results, no concurrency). Computes `collision_loss = (1/(T·D))·Σ‖z−w‖²` over active rows, `logdet_forward`, `logdet_backward`, and `logdet_total = logdet_forward + logdet_backward`.
   - `pub fn midpointCollisionParallel(self: *RSF, input: *const RSFLatentState, target: *const RSFLatentState, allocator: Allocator) !MidpointResult` — same result, with the two frontiers executed on two threads (`std.Thread.spawn`, joined before return; each frontier gets its own scratch from `scratchAllocator()` so no scratch is shared). **Locking contract (mandatory, document it in the doc comment)**: the forward frontier would take per-layer read locks in ascending order and the backward frontier in descending order, which is a deadlock; therefore this function acquires **one model-level read lock for the whole traversal** and both frontiers read weights without taking per-layer locks. `midpointCollision` (sequential) uses the same single model-level read lock so both paths have identical locking semantics. Gate 9(b) must be measured with this concurrent function on CPU and with the single fused kernel on device.
   - `pub fn midpointBackward(self: *RSF, result: *const MidpointResult, grad_scale: f32, logdet_weight: f32, grad_input_out: *RSFLatentState) !void` — seeds `∂ℓ/∂z = 2·(z−w)·grad_scale/(T·D)` and `∂ℓ/∂w = −2·(z−w)·grad_scale/(T·D)`, propagates the first through the forward frontier's adjoint (layers `M−1 … 0`, using `couplingAdjointBatch`) and the second through the backward frontier's inverted-flow adjoint (layers `M … L−1`, using `couplingInvertedFlowAdjointBatch` with `ld_shift = logdet_weight / T`), accumulating into the per-layer gradient tensors via `accumulateLayerGradients`, and writing `grad_input_out`. The two adjoint chains are independent as well; a `midpointBackwardParallel` variant follows the same locking contract as `midpointCollisionParallel`.
     **State recomputation (mandatory — no activation storage anywhere).** The forward-frontier adjoint walks *descending* from `z_M` through layers `M−1 … 0`: at each layer it applies the **inverse** map to recompute that layer's input state `(x1, x2)` and then evaluates the **forward-map** adjoint (`couplingAdjointBatch`) on the recomputed inputs. The backward-frontier adjoint walks *ascending* from `w_M` through layers `M … L−1`: at each layer it applies the **forward** map to recompute that layer's output state and then evaluates the **inverse-map** adjoint (`couplingInvertedFlowAdjointBatch`, Section 4.7) on it. Each adjoint walk therefore costs `⌈L/2⌉` combined recompute+adjoint steps, `midpointCollision` costs `L` applications and `midpointBackward` costs `L` more (`2·L` total), and no intermediate activation of either frontier is ever materialized — this is the same recomputation discipline `rsf_stack_backward_gradients_fused` already uses.
   - **Depth accounting** for gate 9(a): `RSFCore` keeps `layer_applications: std.atomic.Value(usize)` (incremented once per coupling application in every traversal path) and `frontier_depth: std.atomic.Value(usize)` (the longest chain of applications performed by a single frontier during the last midpoint call), with `pub fn resetLayerApplicationCounter(self: *RSF) void` and `pub fn layerApplicationCount(self: *const RSF) usize`, `pub fn lastFrontierDepth(self: *const RSF) usize`. Tests assert: `forward_layers + backward_layers == L`; `max(forward_layers, backward_layers) == ⌈L/2⌉`; `layerApplicationCount` after `midpointCollision` equals `L` and after `midpointBackward` equals `2·L`; `lastFrontierDepth ≤ ⌈L/2⌉`. **State plainly in the doc comments and in the delivery notes that total work is unchanged (2·L coupling applications, the same as a single-frontier forward+backward): the 2× gain is in dependency depth — the longest serial chain drops from `2·L` to `2·⌈L/2⌉` — and it becomes wall-clock only because the two frontiers execute concurrently.**
   - `pub fn resonanceDrift(self: *RSF, state: *const RSFLatentState, allocator: Allocator) !f32` — flows a clone through the stack and records `E = ‖x1‖² + ‖x2‖²` (f64 accumulation) at every `ROTATION_ORDER = 8`-layer boundary and at layer 0; returns `max_l |E_l − E_0| / max(E_0, 1e-30)`. The doc comment must state exactly what Section 4.4 states: this is a bounded diagnostic on the C8 phase cycle, not an exact conservation law, because the coupling factor is not of finite order.
   - `pub fn stateGrowthReport(self: *RSF, state: *const RSFLatentState, allocator: Allocator) ![]f32` — flows a clone through the stack and returns `max|x|` (f32, from an f64 running maximum) at layer 0 and at every `ROTATION_ORDER = 8`-layer boundary; length `1 + ⌈L/8⌉`. This is the depth-conditioning monitor required by Section 4.4 and the input to Phase 13's depth-scaling test; it allocates one clone and no per-layer storage.
9. Tests (under `test-rsf`): all existing tests unchanged and passing (with `global_diffusion = false` where they assert legacy numerics); `RSFLatentState` roundtrip and log-det accumulation; v6 golden load; bound entry points reject wrong-model/wrong-dim states with `RSFBindingError`; `exactSpectralNormRank2` delegation equals the tensor kernel and the layer constraint keeps `σ_max ≤ 0.9·(1+1e-6)`; causal sequence forward/inverse roundtrip `< 5e-7` and all-zero-mask equivalence to the per-token path; midpoint collision on a `L = 4`, `dim = 8` model with `y* = F(x₀)` produced by the ordinary forward yields `collision_loss < 1e-6` (exact automorphism ⇒ exact collision) and `midpointBackward` gradients match `backwardWithLogDet` gradients within rel `1e-3`; layer-application counter gate; `resonanceDrift` finite and monotone-bounded on a spectrally constrained model. **Blocked-case verification (verified constraint):** `test-rsf` is the one spec whose closure reaches `src/hw/accel/accel_interface.zig`, so it cannot link without the generated `main_cpu.c`. Every assertion in this item that does not need the accelerator must therefore also exist in the accel-free `test-rsf-native` spec of Phase 13 (exact rank-2 spectral normalization versus the f64 Jacobi reference, `RSFLatentState` roundtrip and log-det additivity, causal strictness, midpoint depth accounting), so that the phase is verifiable in this environment; `zig build check` must additionally pass for the rewritten `rsf.zig` (it is semantic analysis without linking and is verified to work with no Futhark present). If `test-rsf` itself remains blocked, report it as blocked with the link-time undefined-symbol output attached — not as passed and not as skipped.

## 9. Phase 4 — `src/optimizer/sfd.zig`: the RSF-only optimizer with analytic 2×2 block natural gradient

Rewrite the file completely. Deletions: the private `Tensor`/`Shape`/`TensorFlags`/`Precision`/`erfApprox`/`quantizeValue` stack, `fromCoreTensor`/`toCoreTensor`, `init(allocator, param_size)`, `initWithArena/Pool/Buddy(param_size)` variants, `update`, `updateFusedFisher`, `updateFusedFisherMulti`, `applySlices`, and `accumulateFisher(grads)` in their generic forms. `use_external_fisher` is replaced by explicit Fisher-source control. The scalar diagonal preconditioner is replaced by the analytic block preconditioner of Section 4.6 everywhere.

New contents:

```zig
pub const SFDConfig = struct {
    beta1: f32 = 0.9,            // momentum EMA decay β1
    fisher_gamma: f32 = 0.99,    // the single Fisher/covariance EMA decay γ, used in every FisherMode;
                                 // matches trainer sfd_fisher_gamma_default and is what the host passes
                                 // into the device kernel's `beta2` parameter (Phase 5)
    fisher_epsilon: f32 = 1e-8,  // natural-gradient damping λ; matches sfd_fisher_epsilon_default
    fisher_mode: FisherMode = .rsf_gauss_newton,  // .rsf_gauss_newton | .gradient | .external
    eps: f32 = 1e-8,
    clip_threshold: f32 = 0.1,   // trust ratio τ
    weight_floor: f32 = 1e-3,
    fisher_max: f32 = 1e6,
    warmup_steps: usize = 10,
    spectral_target: f32 = 0.9,  // rsf LAYER_TARGET_SPECTRAL_NORM — exact closed form, no iterations
};

pub const FisherMode = enum { rsf_gauss_newton, gradient, external };

pub const StepStats = struct {
    step: u64,
    lr_effective: f32,
    grad_global_norm: f64,
    fisher_block_mean: f64,       // mean of F_ww and F_bb over all layers/dims
    fisher_offdiag_mean_abs: f64, // mean |F_wb| — measures the coupling the block preconditioner exploits
    correlation_abs_mean: f64,    // mean |ρ_d| = |F_wb|/sqrt(F_ww*F_bb) — bounds the achievable gain (Section 4.6)
    condition_number_max: f64,    // max over d of (λ_max/λ_min) of (F̂_d + λI), f64
    clipped_fraction: f64,
    spectral_reprojected: bool,
};
```

The legacy `beta2 = 0.999` field of the current `SFDConfig` is **removed**: one decay `γ = fisher_gamma` governs every Fisher/covariance update in all modes, matching the trainer's production value and the device kernel. The v1 state loader (magic `0x53464433`) ignores any `beta2` stored in the legacy header and uses the configured `fisher_gamma`; document this in `loadState`.

`SFD` state (all tensors bound; per-layer arrays of length `model.layerCount()`):

- `fisher_blocks_s`, `fisher_blocks_t: []Tensor` — `.fisher_block`, `[dim, 3]` storing `(F_ww, F_wb, F_bb)`
- `momentum_s`, `momentum_t: []Tensor` — `.momentum`, `[dim, 2]`
- `master_s`, `master_t: []Tensor` — `.master_weight`, `[dim, 2]` (FP32 masters; the model weights remain the live copies)
- `model_id`, `dim`, `num_layers`, `step_count: u64`, `cfg`, `allocator`

API:

- `pub fn init(allocator: Allocator, model: *const RSF) !SFD` / `pub fn initWithConfig(allocator: Allocator, model: *const RSF, cfg: SFDConfig) !SFD` — the only constructors; validate config (finiteness, ranges as today plus `0 ≤ fisher_gamma < 1`, `fisher_epsilon ≥ 1e-12`, `0 < clip_threshold ≤ 1`); initialize Fisher blocks and momentum to zero — the damping `λ = fisher_epsilon` in the resolvent keeps the first steps well-defined without special-casing (`A = λ`, `C = λ`, `B = 0` ⇒ `α = 1/λ^{3/2}·…` exactly as the formula yields; verify numerically in a test rather than branching).
- `pub fn deinit(self: *SFD) void`.
- `pub fn step(self: *SFD, model: *RSF, lr: f32) !StepStats` — the complete update:
  1. Validate model identity/dim/layers against `self` (`error.ModelMismatch` otherwise) and `lr` finiteness/positivity.
  2. `model.ensureGradients()`; read per-layer gradients (`readLayerGradients`), gradient products (`readLayerGradientProducts`, branch `s` and `t`), and Gauss–Newton blocks (`readLayerGaussNewton`).
  3. Fisher block update per row `d`, per branch, with `γ = fisher_gamma` and cap `fisher_max` applied to each component:
     - `.gradient` mode: `F_ww ← γF_ww + (1−γ)g_w²`, `F_wb ← γF_wb + (1−γ)g_w g_b`, `F_bb ← γF_bb + (1−γ)g_b²` (exactly the products returned by `readLayerGradientProducts`).
     - `.rsf_gauss_newton` mode: for branch `s`, add the log-det Gauss–Newton block to the empirical block: `F ← γF + (1−γ)(gn_block + grad_products)` where `gn_block[d] = (mean_b x2[d]², mean_b x2[d], 1)`; for branch `t` (zero log-det curvature) it reduces to the `.gradient` update. Document that this mode is RSF-specific by construction — `gn_block` exists only because the RSF log-det is linear in the scale-branch parameters.
     - `.external` mode: blocks are not updated here; the caller supplies them through `setExternalFisher` in the flat layout below.
  4. Analytic block natural-gradient step (Section 4.6), per row `d`, per branch, in this exact order: bias corrections `m̂ = m/(1 − β1^t)`, `F̂ = F/(1 − γ^t)`; `A = F̂_ww + λ`, `B = F̂_wb`, `C = F̂_bb + λ`; `det = max(1e-12, AC − B²)`; `√det`; `S = sqrt(A + C + 2√det)`; `α = 1/(√det·S)`; `inv00 = α(C + √det)`, `inv01 = −αB`, `inv11 = α(A + √det)`; `raw_w = lr_eff·(inv00·m̂_w + inv01·m̂_b)`, `raw_b = lr_eff·(inv01·m̂_w + inv11·m̂_b)`; `lr_eff` includes the existing warmup factor; trust clip each component against `τ·max(|θ|, weight_floor)`; apply `θ ← θ − Δ` to the master weights only when both components are finite; non-finite gradients are treated as zero for momentum (keeping the previous momentum) exactly as in the current code. Then write back through `model.writeLayerWeights`.
  5. Re-constrain the spectral norm per layer to `spectral_target` with `tensor.constrainCouplingSpectralNorm` (single pass, exact); `spectral_reprojected` is true when any rescale exceeded a 1e-6 relative change.
  6. `model.notifyWeightsChanged()`; return measured `StepStats` (global grad norm in f64, block means, mean `|F_wb|`, mean `|ρ_d| = |F_wb|/√(F_ww·F_bb)` — both required by Section 4.6 and by gate 11 —, max condition number computed in f64 from `A + C ± √((A−C)² + 4B²)`, clipped fraction counted over all components). Every field is a real measurement over the tensors just updated; none may be left at a default.
- `pub fn zeroMomentum(self: *SFD) void`, `pub fn resetFisher(self: *SFD) void`.
- `pub fn clipGradNorm(self: *SFD, model: *RSF, max_norm: f32) !f32` — operates on the model's layer gradients through `RSF.scaleLayerGradients`: compute the global norm via `model.gradientL2Norm`, rescale in place when it exceeds `max_norm`, return the pre-clip norm.
- `pub fn ampSchedule(step: usize, warmup: usize, total: usize) f32` — unchanged cosine schedule, free function.
- `pub fn adaptiveLR(grad_norm: f32, param_norm: f32) f32` — unchanged formula, free function.
- `SpectralNormalizer` — keep the type and its API but re-base it on core `Tensor` with binding-aware `normalizeWeights` (validates `.layer_weight_*`/`.master_weight`/`.ranker_head` spaces). For two-column coupling operands it calls `tensor.constrainCouplingSpectralNorm`; its `power_iterations` field and argument are **deleted** for coupling operands and retained only for dense operands of arbitrary column count (which is what the trainer's embedding path needs — Section 4.9 scope rule). `lipschitzRegularization` unchanged.
- `KFACBlock` — redefined RSF-natively and consistently with the block Fisher: `init(allocator, model: *const RSF, layer: usize, damping: f32)` with `A_block: Tensor` `[dim, 3]` over the even-half activations (`(mean x2², mean x2, 1)`, i.e. the Gauss–Newton block of the log-det residual) and `G_block: Tensor` `[dim, 3]` over the odd-half output gradients (`(mean dy2², mean dy1·dy2, mean dy1²)` mapped to the `(w, b)` parameter pair); `updateStatistics(x2_batch: *const Tensor, dy_batch: *const Tensor)` maintaining both with EMA `α`; `preconditionGradient(grad: *Tensor)` applying `G_block^{-1/2} · grad · A_block^{-1/2}` through the same closed-form 2×2 inverse square root used by `step` (one shared internal function `blockInverseSqrt(A, B, C, λ) struct { inv00, inv01, inv11 }` — do not duplicate the resolvent).
- State I/O with two layouts:
  - `pub fn exportFlatState(self: *const SFD, allocator: Allocator) ![]f32` — layout `master_s ‖ master_t ‖ momentum_s ‖ momentum_t ‖ fisher_blocks_s ‖ fisher_blocks_t`; each block is layers in order; master/momentum layers are `dim*2` elements in row-major `[dim, 2]` order identical to the existing trainer checkpoint arrays; Fisher layers are `dim*3` elements in row-major `[dim, 3]` order. This is the layout consumed by `accel.RSFAccelerator.setOptimizerState` (Phase 6) and by the trainer checkpoint (Phase 12).
  - `pub fn importFlatState(self: *SFD, state: []const f32) !void` — validates total length and model identity.
  - `pub fn saveState(self: *const SFD, path: []const u8) !void` / `loadState` — write magic `0x53464434` ("SFD4": header with `model_id`, `dim`, `num_layers`, `step_count`, config fields, `fisher_layout = 3`, then the flat state, CRC32 trailer). `loadState` must also accept the legacy v1 format (magic `0x53464433`, flat `param_size` layout with diagonal Fisher) and promote it deterministically: `F_ww ← fisher_diag[w]`, `F_bb ← fisher_diag[b]`, `F_wb ← 0`, momentum copied per element, masters initialized from the current model weights; reject with `error.InvalidStateFormat` when `param_size` does not match `num_layers·dim·2·2`. Verify with the Phase 0 fixture (d). (There is no shipped diagonal-v2 format: the block layout is the first new format after v1, so no intermediate reader is to be written.)
- `writeBackFP16`/`loadFromFP16` remain as free functions on core `Tensor`.

Distributed trainer integration contract (implemented in Phase 12): the trainer's `fisher_s`/`fisher_t`/momentum/master host arrays are replaced by one `SFD` instance; its checkpoint write/read path calls `exportFlatState`/`importFlatState`; its GPU fused step calls `applyStackGradientsSFD`/`applyUpdateFusedSFD` with hyperparameters taken from `SFDConfig` and Fisher blocks of length `num_layers·half·3`; after any GPU-side state mutation it re-imports device-side state through `getOptimizerState` (added in Phase 6).

Tests (under `test-sfd`, replacing the current 14 tests where their subject is deleted):
- constructor requires an RSF model; config validation;
- **block preconditioner correctness**: for 10,000 PRNG triples `(A, B, C)` with `A, C > 0`, `AC − B² > 0`, assert `‖M^{-1/2} M M^{-1/2} − I‖₂ < 1e-5` where `M = [[A, B], [B, C]]` (f64 reference multiply), and assert `M^{-1/2}` is symmetric positive definite;
- **degeneracy check**: when `B = 0`, the block step reduces exactly to the old diagonal step (`inv00 = 1/√A`, `inv11 = 1/√C`) within `1e-6` — this is the compatibility anchor for the checkpoint promotion;
- convergence: the Phase 0 harness scenario re-run — steps to 50% of initial loss ≤ (baseline steps)/3 (gate 11); loss at step 200 < 60% of loss at step 10 on a dim-8/2-layer model fitting a fixed PRNG target batch;
- invertibility preserved after 100 steps (`roundtripError ≤ 1e-3`);
- Fisher block positivity, `F_wb` sign tracking the gradient correlation, `fisher_max` capping per component;
- all three `FisherMode`s run; `.rsf_gauss_newton` produces `F_ww ≥ mean_b(x2²)·(1−γ)` after one step from zero;
- spectral norm stays ≤ `0.9·(1+5%)` after steps, with no iteration parameter anywhere in the call chain (grep-verified);
- flat-state export/import round-trip; legacy v1 `saveState` fixture loads and promotes; `ModelMismatch` on a different model;
- `KFACBlock` preconditioner matches an explicit dense 2×2 reference on small dims.

## 10. Phase 5 — `src/hw/accel/main.fut` and `src/hw/accel/futhark_kernels.fut`: exact, single-pass, non-isolated device kernels

The device kernels are the transcription of the same canonical layer map (Section 4.5) and the same optimizer algebra (Section 4.6) as the CPU kernels. Every formula below must appear in the delivered Futhark source in full — no `-- TODO`, no omitted cases, no reliance on a host-side fallback to hide a missing kernel.

1. **Analytical rank-2 Gram spectral normalization** (replaces the 30-iteration power method; Section 4.9):
   - Add `let gram_sigma_max_2col [m] (w: [m][2]f32) : f32`. **Accumulate `a`, `b`, `c` in `f64`** (map the sanitized `f32` inputs with `f64.f32`, reduce with `f64.sum`/`reduce (f64.+) 0`, compute the closed form in `f64`, cast the result back with `f32.f64` and guard with `f32.max 0`): at `m = 49152` an `f32` accumulation of the Gram sums carries a relative error of order `1e-5`, which would break the `2.0e-6` agreement gate of Phase 5 item 7 and Phase 14 item 2. Document the accumulation order. Formulas: `a = Σ_i w[i][0]²`, `b = Σ_i w[i][0]·w[i][1]`, `c = Σ_i w[i][1]²`, `tr = a + c`, `diff = a − c`, `Δ = diff² + 4b²`, `λ_max = (tr + sqrt Δ)/2`, `σ_max = sqrt(max(0, λ_max))`.
   - Add `let spectral_normalize_matrix_exact [m] (w: [m][2]f32) (target: f32) : ([m][2]f32, f32, f32)` — sanitize, compute `σ_max`, `scale = if σ_max > max(target, 1e-6) then target/σ_max else 1`, return `(w·scale, σ_max, σ_max·scale)`.
   - Add the entry point with exactly this signature:
     `entry stack_spectral_normalize_exact [layers][rows] (weights: *[layers][rows][2]f32) (target: f32) : (*[layers][rows][2]f32, f32, f32)`
     where the two scalars are `max` σ over layers before and after normalization (matching the existing `stack_spectral_normalize` return convention).
   - **Delete** `entry stack_spectral_normalize` and `let spectral_normalize_matrix`. The execution cost collapses from 30 sequential kernel-internal iterations to one reduction pass; the bindings and the host call site are updated in Phase 6.
   - **Keep** `entry embedding_spectral_normalize` and its ping-pong `u`/`v` power iteration **unchanged**: its operand is a dense `[vocab_size][dim]` matrix for which the 2×2 Gram closed form is mathematically inapplicable (Section 4.9 scope rule). Add a comment at the definition stating exactly this, so the asymmetry is not "fixed" later by mistake.
2. **Exact 2×2 block-diagonal Natural Gradient update** (Section 4.6):
   - Add `let sfd_block_2x2_update (w: [2]f32) (g: [2]f32) (m: [2]f32) (f: [3]f32) (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32) (optimizer_step: i64) (epsilon: f32) (trust_ratio: f32) (weight_floor: f32) : ([2]f32, [2]f32, [3]f32)` implementing, with the same `sanitize_f32`/clamping discipline as `sfd_fisher_update_core`:
     ```
     m_w ← β1·m_w + (1−β1)·g_w ;  m_b ← β1·m_b + (1−β1)·g_b          (β1 = safe momentum_beta)
     F_ww ← γ·F_ww + (1−γ)·g_w² ; F_wb ← γ·F_wb + (1−γ)·g_w·g_b ; F_bb ← γ·F_bb + (1−γ)·g_b²   (γ = safe fisher_gamma, each capped to [0, 1e6])
     m̂ = m / max(eps, 1 − β1^t) ; F̂ = F / max(eps, 1 − γ^t)          (t = max(1, optimizer_step) as f32)
     A = F̂_ww + λ ; B = F̂_wb ; C = F̂_bb + λ                          (λ = max(epsilon, 1e-12))
     det = max(1e-12, A·C − B²) ; √det = sqrt det ; S = sqrt(A + C + 2·√det) ; α = 1/(√det·S)
     Δw = clip( α·((C + √det)·m̂_w − B·m̂_b)·lr , ±τ·max(weight_floor, |w_w|) )
     Δb = clip( α·(−B·m̂_w + (A + √det)·m̂_b)·lr, ±τ·max(weight_floor, |w_b|) )
     w ← w − Δ   (component-wise, only where the result is finite; otherwise keep the old value)
     ```
   - Add the entry point with exactly this signature (the Fisher array is `[layers][rows][3]f32`):
     `entry stack_update_sfd_block2x2_master [layers][rows] (master_weights: *[layers][rows][2]f32) (gradients: [layers][rows][2]f32) (momentum: *[layers][rows][2]f32) (fisher_blocks: *[layers][rows][3]f32) (lr: f32) (beta1: f32) (beta2: f32) (step: i64) (eps: f32) (trust_ratio: f32) (weight_floor: f32) : (*[layers][rows][2]f32, *[layers][rows][2]f32, *[layers][rows][3]f32)`
     Here `beta2` is the Fisher decay `γ` (name kept from the source plan; the host passes `SFDConfig.fisher_gamma`), and the three returned arrays are the updated masters, momentum, and Fisher blocks.
   - **Delete** `entry stack_update_sfd_master`. **Keep** `let sfd_fisher_update_core` and `entry embedding_update_sfd_master` unchanged (the embedding operand is dense `[vocab_size][dim]`; `embedding_update_sfd_master` calls `sfd_fisher_update_core`, verified) and comment the reason at the definition site.
   - Update `futhark_kernels.fut`: `clip_fisher`, `update_fisher`, `compute_natural_grad`, `sfd_fused_step_1d` operate on flat 1-D diagonal state. Replace `compute_natural_grad`'s diagonal division with a block-aware entry `entry block_natural_grad_2x2 [n] (grad_pairs: [n][2]f32) (fisher_blocks: [n][3]f32) (damping: f32) : [n][2]f32` implementing the same resolvent, keep `clip_fisher`/`update_fisher`/`sfd_fused_step_1d` for the dense embedding and 1-D utility paths, and add `entry update_fisher_block [n] (fisher_blocks: [n][3]f32) (grad_pairs: [n][2]f32) (decay: f32) : [n][3]f32`. Any kernel left in the file must have a named caller (grep-verify); delete unreachable ones.
3. **Global diffusion `Θ = Q_r ⊗ H_{2^k}`** (Section 4.5), unparameterized and volume-preserving:
   - `let fwht_stages [n] (x: [n]f32) (block: i64) (stages: i64) : [n]f32` — `loop cur = x for s < stages do` one butterfly stage with `h = 2^s` (compute `h` by `i64` shift), applied inside each contiguous block of length `block`: for global index `i`, `o = i % block`, and `out[i] = if (o / h) % 2 == 0 then (cur[i] + cur[i+h])·inv_sqrt2 else (cur[i−h] − cur[i])·inv_sqrt2`. Guard `i+h < n` structurally by construction (blocks are exact multiples), never by masking values to zero.
   - `let mix_radix_blocks [n] (x: [n]f32) (block: i64) (radix: i64) : [n]f32` — `S[o] = Σ_{b<radix} x[b·block + o]` (a `reduce` over a `tabulate` of block offsets), then `out[i] = x[i] − (2/radix)·S[i % block]`; correct for `radix == 1` (yields `−x`).
   - `let diffuse_row [n] (x: [n]f32) (radix: i64) (block: i64) (stages: i64) : [n]f32 = mix_radix_blocks (fwht_stages x block stages) block radix`.
   - Integrate into the canonical layer step: in `rsf_stack_coupling_row`, after the OFTB rotation `(o1, o2)` and **before** `clamp_f16_value`, apply `diffuse_row` when the `diffusion: bool` argument is true; in `rsf_stack_invert_row`, apply `diffuse_row` to the incoming row **before** the OFTB adjoint `(u1, u2)`. Thread a `diffusion: bool` scalar parameter through `rsf_forward`, `rsf_stack_forward`, `rsf_stack_inverse`, `rsf_stack_backward_gradients_fused`, and the new entries below, and compute `radix`/`block`/`stages` on the host (Phase 6) from `rsfDiffusionLayout(model_dim)` — pass them as `i64` scalars so the kernels stay shape-generic. `Θ` is its own adjoint, so in `rsf_stack_backward_gradients_fused` `diffuse_row` is applied to each incoming activation row **before** computing `u1`/`u2` and to each incoming gradient row **before** computing `h1`/`h2` — the exact mirror position of the forward layer's final `Θ` — with every other per-layer gradient formula unchanged and the log-det accumulation unchanged (the diffusion contributes exactly `0`).
   - Because `H` and `Q_r` are exact and unparameterized, `logdet` accumulation is unchanged: the diffusion contributes exactly `0`.
4. **Dual-frontier symplectic midpoint kernel** (Section 4.7):
   - `entry rsf_stack_midpoint_fused [batch_size][seq_len][half][num_layers] (inputs: [batch_size][seq_len][half*2]f16) (targets: [batch_size][seq_len][half*2]f16) (lengths: [batch_size]i64) (weights_s: [num_layers][half][2]f16) (weights_t: [num_layers][half][2]f16) (clip_min: f32) (clip_max: f32) (logdet_weight: f32) (diffusion: bool) (grad_mean: bool) (gradient_scale: f32) : (*[num_layers][half][2]f32, *[num_layers][half][2]f32, *[batch_size][seq_len][half*2]f16, f32, f32, f32)`
     The parameter list of the source plan is preserved in order; `diffusion`, `grad_mean`, and `gradient_scale` are appended because the device path must match the CPU layer map for both diffusion settings and must reproduce the trainer's gradient normalization. The six returns are: `grad_s` (all layers; layers `0..M−1` from the forward frontier adjoint, layers `M..L−1` from the inverted-flow adjoint), `grad_t` (same partition), `input_delta` (gradient w.r.t. `inputs`, from the forward frontier only, FP16-clamped, zero for inactive tokens via the same `scatter`/`active_indices` discipline as `rsf_stack_backward_gradients_fused`), `collision_loss` (mean squared midpoint mismatch over active tokens and channels, divided by `count_elements` when `grad_mean`), `logdet_forward_mean`, `logdet_backward_mean` (each `Σ clip / valid_tokens` over its own frontier).
     Implementation requirements: `M = num_layers / 2`; `input_delta` derives solely from the collision residual — the midpoint formulation has no separate reconstruction term, which is why `reconstruction_alpha` and the `originals` array are not parameters of this entry (see Phase 6 item 4 and Phase 12 item 1 for how the host maps this onto the existing three-scalar reporting shape). the forward frontier is one `loop` over `l < M` applying `rsf_stack_coupling_row`; the backward frontier is one `loop` over `i < num_layers − M` applying `rsf_stack_invert_row` to the target rows with `l = num_layers − 1 − i`; the two frontier traversals are written as two independent `map`/`loop` nests over the token axis so the Futhark compiler schedules them as independent work (state this intent in a comment; do not serialize one frontier inside the other's loop body). The adjoint stage consists of two more independent loop nests, mirroring Phase 3's state-recomputation rule exactly: (i) a **descending** walk from `z_M` over `l = M−1 … 0` that recomputes each layer's input row with `rsf_stack_invert_row` and applies the forward-map adjoint to it, and (ii) an **ascending** walk from `w_M` over `l = M … L−1` that recomputes each layer's output row with `rsf_stack_coupling_row` and applies the inverted-flow adjoint formulas of Section 4.7 to it, both with `ld_shift = logdet_weight / valid_tokens` and both applying `diffuse_row` at the positions Section 4.5 prescribes (forward-map adjoint: `Θ` last on activations, first on gradients; inverted-flow adjoint: the mirror image). Four independent loop nests total, each at most `⌈num_layers/2⌉` sequential steps. No intermediate trajectory is written to global memory: all per-layer states live in the loop-carried tuples, exactly as `rsf_stack_backward_gradients_fused` does today.
5. **Causal relational bitmask coupling** (Section 4.8):
   - `entry rsf_causal_bitmask_forward [batch][seq_len][half] (x: [batch][seq_len][half*2]f16) (bitmask: [seq_len][seq_len]u8) (weights_s: [half][2]f16) (weights_t: [half][2]f16) (clip_min: f32) (clip_max: f32) (diffusion: bool) : *[batch][seq_len][half*2]f16`
   - `entry rsf_causal_bitmask_inverse [batch][seq_len][half] (y: [batch][seq_len][half*2]f16) (bitmask: [seq_len][seq_len]u8) (weights_s: [half][2]f16) (weights_t: [half][2]f16) (clip_min: f32) (clip_max: f32) (diffusion: bool) : *[batch][seq_len][half*2]f16`
     The source plan's signatures are preserved; `clip_min`, `clip_max`, and `diffusion` are appended because the layer map is undefined without them (`clip_min`/`clip_max` default to the model's `RSFConfig` values on the host).
     Forward: `K_t[d] = Σ_{t'} if bitmask[t][t'] != 0 then X_2,t'[d] else 0` (a masked `reduce` over the sequence axis — branchless in the sense that it is a single reduction with a multiplicative mask, not a per-neighbor `if`), `s_t[d] = clip(f32.f16 weights_s[d][0]·K_t[d] + f32.f16 weights_s[d][1])`, `Y_1,t = X_1,t·exp(s_t)`, `Y_2,t = X_2,t + (f32.f16 weights_t[d][0]·Y_1,t[d] + f32.f16 weights_t[d][1])`, then the OFTB rotation and (when `diffusion`) `diffuse_row`, then `clamp_f16_value`.
     Inverse: the three-step `O(1)`-auxiliary procedure of Section 4.8 — first recover `X_2,t` from `Y` in parallel over `t` (after undoing `diffuse_row` and the OFTB rotation), then re-evaluate `K_t` from the recovered `X_2`, then `X_1,t = Y_1,t·exp(−s_t)`. The kernel must not materialize any `[seq_len][seq_len][half]` intermediate.
   - These entries handle **one layer**. The training path needs the whole stack in a single launch, so also add `entry rsf_causal_bitmask_forward_stack [batch][seq_len][half][num_layers] (x: [batch][seq_len][half*2]f16) (bitmask: [seq_len][seq_len]u8) (weights_s: [num_layers][half][2]f16) (weights_t: [num_layers][half][2]f16) (clip_min: f32) (clip_max: f32) (diffusion: bool) : *[batch][seq_len][half*2]f16` and the matching `rsf_causal_bitmask_inverse_stack`, implemented as a `loop` over layers carrying the row set plus one `K` buffer, with no per-layer round trip to global memory. The single-layer entries remain for tests and for host-orchestrated paths. The masked prefix accumulation is `O(seq_len²·half)` work per layer for a dense mask — report it in the Phase 12 benchmark rather than letting it appear free.
   - Host-side mask validation is mandatory before upload: strictly lower-triangular, values in `{0,1}` (`types.RSFSequenceMask.init` already enforces this; the device path must only ever receive a validated mask, and `accel` re-checks the triangularity of the uploaded bytes once per mask).
6. **Legacy variant audit** (mandatory, grep-verified): `rsf_scatter`, `rsf_flow`, `rsf_flow_logdet`, `rsf_invert_flow`, `rsf_forward_layer`, `rsf_forward_multi`, `entry rsf_forward_multilayer`, `rsf_backward_scatter`, `rsf_backward_flow` in `futhark_kernels.fut` implement a permutation-parameterized coupling variant that does **not** apply the OFTB rotation and is not on the stack training path. For each one: list its callers (in `.fut` sources and in `futhark_bindings.zig`/`accel_interface.zig`). If it has no live caller, delete it. If it has one, either bring it to the canonical layer map of Section 4.5 (rotation + diffusion, with the permutation retained as an additional fixed factor and its log-det contribution proven to be 0) or keep it and document at the definition site, in one sentence, exactly which host path uses it and why it is exempt. Silent divergence between two device layer maps is a defect. `rgpu_hadamard_transform`/`rgpu_hadamard`/`rgpu_hadamard_batch` are the complex single-qubit Hadamard gate of the quantum path; they are unrelated to the real normalized FWHT of item 3 (different domain, different normalization) and must not be reused for `Θ`.
7. Futhark-level verification (run when the Futhark toolchain is present; otherwise report as unverified, per Section 3 constraint 1): `futhark check`/the repository's `zig build futhark-test` path compiles both files; a Futhark-level test (or the host test in Phase 6) asserts `diffuse_row (diffuse_row x r b s) r b s == x` within `1e-6` for `(r, b, s) = (3, 32768, 15)` and `(1, 32, 5)`; `gram_sigma_max_2col` matches the CPU `exactSpectralNormRank2` within `2e-6` on the same PRNG matrices; `sfd_block_2x2_update` with `B = 0` reproduces `sfd_fisher_update_core` within `1e-5`.

## 11. Phase 6 — `src/hw/accel/futhark_bindings.zig`, `src/hw/accel/accel_interface.zig`, `src/hw/accel/gpu_memory.zig`

1. **Bindings.** The authoritative source of every declaration is the C header that Futhark generates from the Phase 5 sources; regenerate it and transcribe it — never hand-write a signature and hope it links. **Known shape rule:** because `stack_spectral_normalize_exact` returns an *in-place* array (`*[layers][rows][2]f32`) together with two scalars, Futhark emits an **opaque tuple handle plus `futhark_project_opaque_tup3_*` accessors** — exactly as today's `stack_spectral_normalize` does with `struct_futhark_opaque_tup3_stack_spectral` — and *not* three separate `out0`/`out1`/`out2` parameters. The listing below is reproduced from the source plan; wherever it disagrees with the generated header, **the generated header wins**, and the difference is recorded in the delivery notes. If the three-out-parameter form is wanted, the `.fut` entry must return fresh (non-in-place) arrays instead, and `accel_interface.zig` must then allocate and own the returned array — pick one form deliberately, never leave both.
   - New opaque tuple type and its accessors:
     ```zig
     pub const struct_futhark_opaque_tup3_stack_sfd_block2x2 = opaque {};

     pub extern "c" fn futhark_entry_stack_spectral_normalize_exact(
         ctx: ?*struct_futhark_context,
         out0: ?*?*struct_futhark_f32_3d,
         out1: ?*f32,
         out2: ?*f32,
         weights: ?*const struct_futhark_f32_3d,
         target: f32,
     ) c_int;

     pub extern "c" fn futhark_entry_stack_update_sfd_block2x2_master(
         ctx: ?*struct_futhark_context,
         out: ?*?*struct_futhark_opaque_tup3_stack_sfd_block2x2,
         master_weights: ?*const struct_futhark_f32_3d,
         gradients: ?*const struct_futhark_f32_3d,
         momentum_state: ?*const struct_futhark_f32_3d,
         fisher_blocks: ?*const struct_futhark_f32_3d,
         learning_rate: f32,
         momentum_beta: f32,
         fisher_gamma: f32,
         optimizer_step: i64,
         epsilon: f32,
         trust_ratio: f32,
         weight_floor: f32,
     ) c_int;

     pub extern "c" fn futhark_free_opaque_tup3_stack_sfd_block2x2(
         ctx: ?*struct_futhark_context,
         obj: ?*struct_futhark_opaque_tup3_stack_sfd_block2x2,
     ) c_int;

     pub extern "c" fn futhark_project_opaque_tup3_stack_sfd_block2x2_0(
         ctx: ?*struct_futhark_context,
         out: ?*?*struct_futhark_f32_3d,
         obj: ?*const struct_futhark_opaque_tup3_stack_sfd_block2x2,
     ) c_int;
     pub extern "c" fn futhark_project_opaque_tup3_stack_sfd_block2x2_1(
         ctx: ?*struct_futhark_context,
         out: ?*?*struct_futhark_f32_3d,
         obj: ?*const struct_futhark_opaque_tup3_stack_sfd_block2x2,
     ) c_int;
     pub extern "c" fn futhark_project_opaque_tup3_stack_sfd_block2x2_2(
         ctx: ?*struct_futhark_context,
         out: ?*?*struct_futhark_f32_3d,
         obj: ?*const struct_futhark_opaque_tup3_stack_sfd_block2x2,
     ) c_int;

     pub extern "c" fn futhark_entry_rsf_stack_midpoint_fused(
         ctx: ?*struct_futhark_context,
         out: ?*?*struct_futhark_opaque_tup6_fused_stack_gradients,
         inputs: ?*const struct_futhark_f16_3d,
         targets: ?*const struct_futhark_f16_3d,
         lengths: ?*const struct_futhark_i64_1d,
         weights_s: ?*const struct_futhark_f16_3d,
         weights_t: ?*const struct_futhark_f16_3d,
         clip_min: f32,
         clip_max: f32,
         logdet_weight: f32,
         diffusion: bool,
         grad_mean: bool,
         gradient_scale: f32,
     ) c_int;
     ```
   - Plus the causal entries (`futhark_entry_rsf_causal_bitmask_forward` / `_inverse`) with their `struct_futhark_u8_2d` handle type and the matching `futhark_new_u8_2d` / `futhark_values_u8_2d` / `futhark_free_u8_2d` accessors if the generated header provides them; if the generated header names the handle differently, use the generated name and record the difference in the delivery notes — never invent a symbol.
   - **Remove** `futhark_entry_stack_spectral_normalize`, its `power_iters` parameter, `struct_futhark_opaque_tup3_stack_spectral` (unless the generated header still emits it for another entry), and `futhark_entry_stack_update_sfd_master` with `struct_futhark_opaque_tup3_stack_sfd` if and only if no remaining entry produces that tuple type. Every removal must be grep-verified against `accel_interface.zig`. Every removal must be grep-verified against `accel_interface.zig`, and the regenerated header must be diffed against the committed one so that no stale declaration survives.
2. **`RSFAccelerator` spectral normalization.** Replace the iterative `spectralNormalizeLayers(self, target, iterations)` with `spectralNormalizeLayers(self: *Self, target: f32) AccelError!void` calling `futhark_entry_stack_spectral_normalize_exact` once per stack (s and t). Delete the ping-pong `u`/`v` device vectors, their allocation, their upload, and every `iterations` argument in this path. Keep the dense embedding normalization (`spectralNormalizeEmbedding` / whatever the current embedding-facing method is named) iterative and untouched, with the Section 4.9 scope comment.
3. **`RSFAccelerator` Fisher state becomes block-shaped.**
   - `RSFOptimizerState` fields: `master_weights_s/t: []f32` and `momentum_s/t: []f32` of length `num_layers·half·2`; `fisher_blocks_s/t: []f32` of length `num_layers·half·3`; `step: u64`; `allocator`. `empty`, `deinit`, `clone`, and every length validation updated accordingly.
   - `setOptimizerState(master_weights_s, master_weights_t, momentum_s, momentum_t, fisher_blocks_s, fisher_blocks_t, step)` — validate lengths (`num_layers·half·2` and `num_layers·half·3`), upload the Fisher arrays as `[num_layers][half][3]f32` device arrays. Add `pub fn getOptimizerState(self: *Self, allocator: Allocator) AccelError!RSFOptimizerState` reading the device arrays back in the same layout (this is the counterpart SFD's `importFlatState` consumes).
   - Add `pub fn promoteDiagonalFisher(allocator: Allocator, fisher_s: []const f32, fisher_t: []const f32) AccelError!struct { blocks_s: []f32, blocks_t: []f32 }` implementing the deterministic promotion of Section 4.6 (`F_ww ← diagonal w entry`, `F_bb ← diagonal b entry`, `F_wb ← 0`), used by the checkpoint legacy reader in Phase 12 and tested against the Phase 0 fixture (e).
   - `applyStackGradientsSFD` / `applyUpdateFusedSFD` / `applyGradientsSFD`: switch to `futhark_entry_stack_update_sfd_block2x2_master`, keep the per-layer `[dim, 2]` gradient stack validation, and keep the existing context-lock discipline (`ctx.mutex`, `requireOwnerLive`, `checkInvariantsUnlocked`).
4. **`fusedMidpointTrainingStep`.** Add:
   ```zig
   pub fn fusedMidpointTrainingStep(
       self: *Self,
       inputs: *FutharkArray3DF16,
       targets: *FutharkArray3DF16,
       sequence_lengths: []const usize,
       grad_mean: bool,
       gradient_scale: f32,
       logdet_weight: f32,
   ) AccelError!FusedStepResult
   ```
   - Bind the forward-half weight stacks `[0 … M−1]` and the backward-half stacks `[M … L−1]` from the already-uploaded FP16 weight arrays (`M = num_layers / 2`); pass whole `[num_layers][half][2]` arrays and let the kernel partition them (no host-side slicing copies, no host synchronization).
   - Call `futhark_entry_rsf_stack_midpoint_fused` once. Project the 6-tuple; keep the gradient arrays and `input_delta` **on device** and feed them directly into the block SFD update and the FP16 writeback — zero host synchronization between the kernel and the optimizer step (no `valuesFlat` in the hot path; scalars are read once at `finalize`).
   - Return the existing `FusedStepResult`/`FusedStepScalars` shape with `loss = collision_loss`, `reconstruction_loss = collision_loss`, `logdet_mean = logdet_forward_mean + logdet_backward_mean`. The doc comment must state why the two losses are the same number: in the midpoint formulation the collision residual `‖z_M − w_M‖²` *is* the prediction and reconstruction residual simultaneously, because `F ∈ Aut(AD)` makes `z_M = w_M` equivalent to `F(x₀) = y*`. Add `collision_loss`, `logdet_forward_mean`, `logdet_backward_mean` as explicit fields on a new `MidpointStepScalars` carried alongside `FusedStepScalars` in the result so nothing is conflated for logging.
5. **Causal sequence entries on device.** Add `causalSequenceForward(self: *Self, state: *FutharkArray3DF16, mask: *const types.RSFSequenceMask, layer: usize) AccelError!void` and `causalSequenceInverse(...)`, uploading the validated mask once per mask instance (cached by identity, invalidated on mask change), re-checking strict lower triangularity of the uploaded bytes, and calling the Phase 5 entries with `clip_min`/`clip_max`/`diffusion` taken from the model configuration passed at accelerator construction (`RSFAccelerator` gains `clip_min`, `clip_max`, `diffusion_enabled`, `diffusion_radix/block/stages` fields set by `initMultiLayer*`; extend those initializers with the values rather than adding a second constructor). The mask is uploaded in the `[seq_len][seq_len]u8` form produced by `types.RSFSequenceMask.toBytes` (the packed `words` form is the storage of record); allocate the byte form per upload, free it after, and never cache a second copy of the mask.
6. **CPU/GPU equivalence gates** (extend `validateAcceleratorAgainstCPU` in `rsf.zig` and the accelerator's self-tests): with diffusion enabled and disabled, device forward vs CPU forward within `MODEL_CROSS_CHECK_ABS_TOL/REL_TOL = 5e-2` (FP16 device arrays); device `stack_spectral_normalize_exact` vs `tensor.constrainCouplingSpectralNorm` within `1e-5` (both are exact — a wider disagreement is a defect, not a tolerance issue); device midpoint gradients vs CPU `midpointBackward` within `5e-2`; device causal forward/inverse vs the CPU causal kernels within `5e-2`, and device causal roundtrip within `1e-5`. Where no device exists, these gates are compiled out by the existing `accel.gpu_enabled` comptime switch and reported as unverified.
7. **`src/hw/accel/gpu_memory.zig`.** `CouplingLayout` keeps `.diagonal` (2 columns) and `.dense_affine`; add `fisher_block: bool = true` to `EstimateConfig`, `fisher_elements_per_stack: u64` and `fisher_bytes_*` to `Estimate`, and size the Fisher stacks at 3 columns per row while masters and momentum stay at 2 (`stackElements` gains a column-count parameter or a sibling `fisherStackElements`). Update both internal `.coupling = .diagonal` sites and the `test-gpu-memory` suite; admission must remain truthful — under-reporting device memory to fit a configuration is a defect.

## 12. Phase 7 — `src/index/ssi.zig`: the RSF latent index

Rewrite while preserving the tree mechanics (height cap, bucketing, collision lists, `mixHash`, compaction, merge/split/balance) and serialization discipline.

1. `Segment` gains: `latent: []f32` (length `2*dim`, the row state even‖odd), `log_det: f32`, `latent_version: u32` (0 = absent, 1 = current layout). `Segment.init` requires the latent; `deinit` frees it.
2. The index records the **layer-map identity** of the model that produced its latents at first insert: `model_id: u64`, `dim: usize`, `global_diffusion: bool`. Retrieval with a query latent produced under a different setting is rejected with `error.RSFModelMismatch` — cosine similarity between a diffused and a non-diffused latent is not meaningful, and silently mixing them is a defect.
3. Signature scheme (deterministic, documented):
   - Token MinHash 64-bit signature: unchanged algorithm.
   - Latent SimHash 64-bit: bit `i` = `1` iff `Σ_j r[i][j]·latent[j] ≥ 0`, where `r[i][j]` are pseudo-random ±1 values from `splitmix64(seed = 0x51DE_0BAA ^ i * 0x9E37_79B9_7F4A_7C15 ^ j)` — implement `latentSimHash(latent: []const f32) u64` and share it with the Ranker through `src/core/types.zig` (`pub fn rsfLatentSimHash(latent: []const f32) u64`) so both modules use one implementation. Because `Θ` mixes every coordinate into every block, the SimHash of a diffused latent is a global feature — state this in the doc comment (it is the reason diffusion improves index selectivity).
   - Combined `signature = mixHash(token_signature, latent_simhash)`.
4. Insertion API (all require latents; token-only insertion no longer exists):
   - `pub fn addSequenceWithLatent(self: *SSI, tokens: []const u32, position: u64, is_anchor: bool, latent: []const f32, log_det: f32) !void`
   - `pub fn addLatent(self: *SSI, model: *const RSF, state: *const RSFLatentState, tokens: []const u32, position: u64, is_anchor: bool) !void` — reads dim and latent from the state, validates the state binding, and records the layer-map identity of item 2.
5. Retrieval:
   - `pub fn retrieveTopKLatent(self: *const SSI, query_latent: []const f32, query_log_det: f32, k: usize, allocator: Allocator) ![]types.RankedSegment` — candidates scored by `score = 0.7·cos + 0.3·volume`, `cos = ⟨q,s⟩/(‖q‖·‖s‖ + 1e-12)`, `volume = exp(−|log_det_s − query_log_det|/κ)` with `κ = 1.0` (both constants named in a `pub const SSIScoringConfig` with these defaults). Because `Θ` and `R` are orthogonal, `‖·‖₂` and therefore `cos` are invariant under the layer map's orthogonal factors — document this as the reason the score is comparable across depths. Segments with `latent_version == 0` (legacy-loaded) fall back to `score = token_signature_similarity` normalized to [0,1] via the existing `signatureSimilarity`.
   - `retrieveTopK(query_tokens, k, allocator)` is re-based on the same traversal and documented as the token-approximation fast path.
   - `updateScore`, `getSegment`, `compact`, `merge`, `split`, `balance`, `stats`, `validate` — preserved, extended to handle latent payloads (`merge` concatenates and re-buckets, refusing to merge indexes with different layer-map identity; `split` partitions by score threshold as today).
6. Tensor I/O: `exportToTensor` emits a `[segments, 2*dim + 4]` tensor carrying an `.index_payload` binding (latent ‖ score, log_det, position_lo, position_hi as f32 bits) plus the header tensor; `importFromTensor` validates and rebuilds. `serialize`/`deserialize`: bump the internal format version; write `dim`, `model_id`, `global_diffusion`, then per-segment latent+log_det+latent_version; the deserializer accepts the old format (latents absent → `latent_version = 0`, zero-length latent, `global_diffusion = false`). Test against the Phase 0 fixture (b).
7. Tests: insert/retrieve with synthetic latents (construct a small RSF, embed fixed token batches, forward them, index, retrieve by a held-out latent — expect the nearest-by-cosine segment in the top-3 for ≥ 9 of 10 queries on a fixed seed, with diffusion both enabled and disabled); legacy serialization loads; merge/split/compact with latents; `RSFModelMismatch` enforcement for foreign model id *and* for a differing `global_diffusion`; export/import round-trip.

## 13. Phase 8 — `src/ranker/ranker.zig`: the RSF-native ranker

1. Delete `vectorScore`, `dotProductScore`, and the raw-tensor scoring path. Keep the n-gram/LSH/MinHash machinery (signatures, Jaccard estimators, streaming, parallel scoring, top-k heap) as the token-prior components.
2. `types.RankedSegment` (in `types.zig`) gains `latent_similarity: f32 = 0`, `reconstruction_confidence: f32 = 0`, `volume_surprise: f32 = 0`; `deinit` unchanged (no new allocations).
3. Fused scoring (exact): for a candidate segment `s` and query latent state `q`:
   - `z_token = sigmoidScale(existing token score)` (existing `RankerConfig.SCORE_SIGMOID_*` transform),
   - `z_latent = cosine(q.latent, s.latent)` clamped to [0,1],
   - `z_recon = clamp(1 − roundtrip_relative_error, 0, 1)` where the error is `RSFLatentState.roundtripError` of the candidate latent through the model (which now includes the diffusion factor — a diffused latent that fails to invert is genuine evidence against the candidate),
   - `z_volume = exp(−|s.log_det − q.log_det|/κ)` with the same κ constant as SSI,
   - `raw = w_token·z_token + w_latent·z_latent + w_recon·z_recon + w_volume·z_volume`, `score = sigmoidScale(raw·RankerConfig.MAX_RAW_SCORE/10)`.
4. The weight vector is an RSF coupling head, not a plain array: `RankerHead` is a one-layer `RSF` model with `dim = 2` over the four criteria packed as a 4-element state `(z_token, z_latent, z_recon, z_volume)`, constructed with `RSFConfig{ .global_diffusion = false, .clip_min = -5.0, .clip_max = 5.0 }` — diffusion is disabled for the head deliberately and documented: the head's readout is the even-half sum of a 4-element state whose coordinates are named criteria, and mixing them with `Θ` would destroy the readout's interpretability while adding no capacity (the head has 4 parameters). `pub fn headForward(self: *Ranker, z: [4]f32) !f32` flows the packed state through the head's coupling layer and OFTB rotation and reads out the even-half sum; the head is owned by `Ranker` (`head: RSF`, `head_sfd: SFD`), initialized with deterministic seeds, and is the only scorer whose parameters are trained.
   - `Ranker.init(allocator, num_ngrams, num_hash_funcs, seed, model: *const RSF)` — the base model supplies `model_id`/`dim`/`global_diffusion` for latent compatibility; `Ranker` refuses to score latents from another model or another layer map (`error.RSFModelMismatch`).
5. `calibrateWeights` → `pub fn trainHead(self: *Ranker, training_data: []const []const u32, labels: []const f32, ssi: *const SSI, model: *RSF, epochs: usize) !void`: builds per-example criteria vectors from SSI retrieval against each training sequence, forms the pairwise logistic ranking loss `L = Σ log(1 + exp(−(score_pos − score_neg)))` over label-ordered pairs (adjacent pairs suffice; document the pairing rule), backpropagates through the head via the head model's own `backward` on the 4-element state, and updates head parameters with the owned `SFD` (which now applies the analytic 2×2 block preconditioner to the head's `[2, 2]` coupling matrices — for `dim = 2` that is two 2×2 blocks, i.e. the whole head is preconditioned exactly). This replaces hand-rolled SGD entirely; `updateWeights(gradients: []const f32)` is deleted.
6. `rankCandidates`/`rankCandidatesWithQuery` gain model-required forms: `rankCandidatesWithModel(self, model: *const RSF, query_state: *const RSFLatentState, candidates: []types.RankedSegment, ssi, allocator) !void` — fills the three new `RankedSegment` fields and the fused score. The old signatures are removed. `topKHeap`, `streamingRank`, `parallelScore`, `batchScore` are re-based on the fused score (streaming/parallel keep their buffer/thread mechanics; thread counts and chunking unchanged).
7. `exportModel`/`importModel`: version bump; serialize n-gram/LSH state as today plus the head coupling weights, the head's `global_diffusion` flag, and the head SFD state (block Fisher); the importer accepts legacy files (no head section) by initializing a fresh default head deterministically from the stored seed, and accepts an SFD v1 state through the Phase 4 promotion path. Test with the Phase 0 fixture (c).
8. Tests: fused-score ordering matches a reference implementation of the formulas on fixed inputs; head training reduces the pairwise ranking loss over 50 epochs on a synthetic labeled set (deterministic seed, loss strictly decreases between epoch 1 and epoch 50); `RSFModelMismatch` on foreign latents and on a diffusion mismatch; streaming/parallel equality with the sequential path on fixed data; legacy import; top-k heap correctness with the new fields.

## 14. Phase 9 — `src/core_relational/vpu.zig`: the RSF vector execution unit

1. New first-class lane type: `pub fn CouplingLanes(comptime N: usize) type { return struct { even: @Vector(N, f32), odd: @Vector(N, f32) } }` with `F32Coupling8 = CouplingLanes(8)` and an AVX-512 16-lane variant selected by the same runtime detection pattern as `OFTB.vectorLen()`.
2. Coupling lane ops mirroring the canonical algebra (call the `tensor.zig` kernels on lane-sized scratch or implement `comptime`-specialized lane forms that are tested equivalent to the tensor kernels — the equivalence test is mandatory either way): `couplingForwardLanes`, `couplingInverseLanes`, `couplingAdjointLanes` (returns `dx` lanes and accumulates parameter grads into caller-provided `[dim, 2]` slices), `couplingInvertedFlowAdjointLanes` (the Section 4.7 adjoint, used by the midpoint path), `logDetLanes` (f32 reduce-add), and two diffusion helpers: `butterflyStagePairLanes(lanes: *CouplingLanes(N))` — the single 2-point stage `(even, odd) ↦ ((even+odd)/√2, (even−odd)/√2)`, which is exactly the `h = 1` butterfly restricted to two adjacent lanes and **not** a full transform (say so in the doc comment) — and `diffuseRowLanes(row: []f32, layout: types.RSFDiffusionLayout) !void`, the full `Θ` on a row held in lane-tiled scratch, delegating to `tensor.diffuseRowInPlace` so there is one butterfly implementation in the system.
3. `VPU` facade changes:
   - `VectorBatch` storage is re-typed to batches of coupling lanes plus their bindings; `processVectors` accepts new `BatchOperation` variants `.coupling_forward`, `.coupling_inverse`, `.coupling_adjoint`, `.coupling_diffuse`, `.coupling_causal_forward` (parameters supplied via a `CouplingParamsRef { model_id, layer_index, s_weight: *const Tensor, t_weight: *const Tensor, clip_min, clip_max, diffusion: bool, mask: ?*const types.RSFSequenceMask }` validated through tensor bindings). The existing arithmetic `BatchOperation` variants remain for the quantum/LNS utilities that consume them.
   - `batchMatmul`, `powerIteration`, `computeSimilarityMatrix`, `quantumVectorOps`, `Matrix4x4`/`MatrixOps`, `LNSValue`/`LNSInstruction`, `SimdVector` library: retained (they serve `quantum_logic`, `ibm_quantum`, `chaos_core`), but `computeSimilarityMatrix` and `powerIteration` are re-based to accept `CouplingLanes`/bound-tensor inputs in addition to their current forms where the current forms are consumed by tests only — keep every symbol that other modules import (grep first). `powerIteration` remains for *diagnostics on arbitrary dense operands*; it must not appear in any coupling normalization path (Section 4.9), and the doc comment must say so.
   - `computeGraphEmbeddings` is replaced by `pub fn computeGraphLatents(self: *VPU, graph: *SelfSimilarRelationalGraph, model: *const RSF, allocator: Allocator) !ArrayList(RSFLatentState)`: each node's `data` bytes are hashed deterministically (FNV-1a over bytes, then `splitmix64`) into `2*dim` floats scaled to `[-1,1]`, packed as a one-row `RSFLatentState`, and flowed through `model` via `forwardLatentWithLogDet`; edge weights are updated to the cosine of the endpoint latents through the existing `SelfSimilarRelationalGraph` edge API. `buildAdjacencyBitmask` is kept and extended: `BitmaskMatrix.initFromGraph` unchanged, plus `pub fn latentChannelMask(latents: []const RSFLatentState, threshold: f32, allocator) !BitmaskMatrix` producing a `[nodes, 2*dim]` bitmask of channels whose `|value| > threshold`.
   - **Relational causal masks** (this is what makes cross-token coupling relational rather than merely sequential):
     - `pub fn causalMaskFromSequence(allocator: Allocator, seq_len: usize) !types.RSFSequenceMask` — the full strictly lower-triangular mask.
     - `pub fn relationalCausalMask(allocator: Allocator, graph: *const SelfSimilarRelationalGraph, node_order: []const []const u8) !types.RSFSequenceMask` — builds `C[i][j] = 1` iff `j < i` **and** the graph has an edge between `node_order[j]` and `node_order[i]` (using `exportAdjacencyBitmask`), i.e. a token only receives context from tokens it is relationally connected to. Validate that every id in `node_order` exists in the graph (`error.NodeNotFound`); assert strict lower triangularity before returning. Document and test the degenerate case: when no edge lies below the diagonal, `nnz = 0`, every `K_t = 0`, and the causal layer reduces exactly to the per-token coupling with `s_t = clip(b_s)` constant across tokens — assert that equality in a test instead of treating zero density as an error.
     - `pub fn causalMaskDensity(mask: *const types.RSFSequenceMask) f32` — measured fill ratio, reported in `VPUStatistics`.
   - `propagateBitmaskSignal`/`booleanPropagateBitmask`: the signal-propagation math moves onto the bit-plane popcount implementation of Phase 11 (`nsir_core.bitmaskSignalPropagate`), with `nsir_core.bitmaskSignalPropagateExact` retained as the reference; document and test them as scatter-flow scheduling masks. Add `pub fn scatterFlowPropagate(self: *VPU, state: *RSFLatentState, mask: *const BitmaskMatrix, row: usize, model: *RSF, layers: usize) !void` — applies `layers` coupling forwards where the mask's row bits are set and identity where cleared (per-channel bypass; the mask must have `2*dim` columns for the state's dim; validated).
4. `MemoryPool`, `VectorCache`, `VPUStatistics`: preserved; statistics gain counters `coupling_ops`, `latents_flowed`, `diffusion_rows`, `causal_masks_built` (real measured counts).
5. Tests (under `test-vpu`): lane-op equivalence to `tensor.zig` kernels including `diffuseLanesBlock` vs `walshHadamardInPlace` (PRNG sweep, abs 1e-6 / rel 1e-5); inverse roundtrip through lanes ≤ 1e-4; `computeGraphLatents` produces finite latents with correct bindings and log-dets matching `model.meanLogDetJacobian` within 1e-3 on a small graph; `relationalCausalMask` yields a strictly lower-triangular mask whose density equals the measured edge density of the induced subgraph, and flowing a sequence through `RSF.forwardSequence` with a relational mask differs from the full causal mask exactly on the token pairs with no edge; `scatterFlowPropagate` equals full forward when all bits set and identity when none set; `propagateBitmaskSignal` matches `nsir_core.bitmaskSignalPropagateExact` within the Phase 11 quantization bound; existing VPU tests that remain applicable still pass; every deleted symbol is confirmed unused by grep across `src/`.

## 15. Phase 10 — `src/core_relational/r_gpu.zig`: the RSF sharded execution fabric

1. `ProcessingCore` gains `rsf_shard: ?Shard` where
   `Shard = struct { model_id: u64, layer_index: usize, dim_range: [2]usize, s_weight: Tensor, t_weight: Tensor, s_grad: ?Tensor, t_grad: ?Tensor, logdet_contribution: f64, active_channels: usize, diffusion: types.RSFDiffusionLayout, local_stages: usize }`
   (weights are bound copies `.layer_weight_s`/`.layer_weight_t`; grads `.gradient`).
2. **Diffusion-aware partitioning** (mandatory — `Θ` mixes all `row_len` coordinates, so a naive contiguous `dim` split would make shard-local execution impossible):
   - **Partition rule (normative — the only one that makes shard-local butterflies possible).** A core owns coupling indices `d ∈ [c·s, (c+1)·s)`, which in the latent row of length `D = 2·dim` are the two intervals `[c·s, (c+1)·s)` (its `x1` lanes) and `[dim + c·s, dim + (c+1)·s)` (its `x2` lanes). Both intervals of an index must live on the same core, because `C` needs `x1[d]` and `x2[d]` together. When diffusion is enabled and more than one core is used, set `s = dim / c'`, where `c'` is the **largest divisor of `dim` with `c' ≤ cores` such that `s` is a power of two and `s ≤ block/2`**; cores `c' … cores−1` stay idle for that layer and are counted in `RPGUStatistics.diffusion_idle_cores`. With that choice every core interval lies inside a single diffusion block, so butterfly stages `h < s` are core-local while stages `s ≤ h < block` cross cores: `stages − log2(s)` exchange rounds plus one `Q_r` reduce/broadcast round per layer. Worked example for the default geometry (`dim = 49152 = 3·2^14`, `block = 32768`, `stages = 15`) on 8 cores: `c' = 6` (because `dim/6 = 8192 = 2^13 ≤ 16384`, whereas `c' = 12 > 8`), hence `s = 8192`, 2 idle cores, `15 − 13 = 2` butterfly exchanges plus 1 `Q_r` round = **3 exchange rounds per layer**. When `dim/cores` is already an admissible power of two, `c' = cores` and no core idles. When diffusion is disabled, `s = ⌈dim/cores⌉` (the pre-upgrade contiguous split) and no exchange rounds are generated at all.
   - Add `MessageType.rsf_diffuse_exchange` (butterfly stage crossing a core boundary: each core sends its `h`-partner lanes to the owning neighbour and receives its own) and `MessageType.rsf_diffuse_sum` / `.rsf_diffuse_broadcast` (the `Q_r` factor: per-offset partial sums reduced across the `r` block-owners, then the correction `−(2/r)·S[o]` broadcast back). Every exchange is counted in `RPGUStatistics.diffusion_exchanges` and its accumulated latency in `RPGUStatistics.diffusion_cycles` — real measured counts, never estimates.
   - When `diffusion` is disabled (v6-loaded models) no exchange messages are generated at all; this must be asserted in a test.
3. `RelationalGraphProcessingUnit` gains:
   - `pub fn distributeRSFModel(self: *RelationalGraphProcessingUnit, model: *const RSF) !void` — diffusion-aligned partitioning per item 2, round-robin over layers, copying weights through the Phase 3 read surface; clears any previous distribution deterministically.
   - `pub fn forwardRSF(self: *RelationalGraphProcessingUnit, model: *const RSF, state: *RSFLatentState) !void` and `pub fn inverseRSF(...)` — per layer, in layer order (inverse: reverse order): each core computes scale/translation on its dim range for every batch row using the `tensor.zig` row kernels on its shard; the OFTB rotation is lane-local; the diffusion factor executes via the exchanges of item 2; NoC messages (`.rsf_forward_packet`, `.rsf_inverse_packet`, `.rsf_reduce_logdet`, `.rsf_grad_packet`) carry the even/odd half updates to the owning neighbour per the existing routing rules; a per-layer reduce accumulates `logdet_contribution` per core and updates `state.log_det`. The assembled state must satisfy `max |sharded_forward − model.forward| ≤ abs 1e-4 + rel 1e-3·|ref|` elementwise (f32 accumulation-order differences tolerated; test gate in Phase 14).
   - `pub fn forwardRSFSequence(self: *RelationalGraphProcessingUnit, model: *const RSF, state: *RSFLatentState, mask: *const types.RSFSequenceMask) !void` and `pub fn inverseRSFSequence(...)` — the causal coupling of Section 4.8 sharded over dim ranges: the prefix accumulation `K_t[d]` is per-channel and therefore entirely core-local (each core owns its channels for all tokens); the mask is broadcast once to every core and validated on receipt; `logdet` reduces as `Σ_t Σ_d s_{t,d}` across cores.
   - `pub fn backwardRSF(self: *RelationalGraphProcessingUnit, model: *const RSF, grad_output: *const RSFLatentState, input: *const RSFLatentState, output: *const RSFLatentState, grad_input_out: *RSFLatentState, logdet_weight: f32) !void` — adjoint in reverse layer order; shard-local grads accumulate in `Shard.s_grad/t_grad`; the diffusion adjoint reuses the same exchange pattern (`Θ^T = Θ`); a final gather writes the summed parameter grads into the model layer gradients via `RSF.accumulateLayerGradients` (added in Phase 3) and produces `grad_input_out`.
   - `pub fn midpointCollisionRSF(self: *RelationalGraphProcessingUnit, model: *const RSF, input: *const RSFLatentState, target: *const RSFLatentState, allocator: Allocator) !RSFMidpointShardResult` — the two frontiers of Section 4.7 execute on **two disjoint core groups** (cores `[0, grid/2)` run the forward frontier over layers `0..M−1`; cores `[grid/2, grid)` run the backward frontier over layers `M..L−1`), which is where the 2× sequential-depth reduction becomes structural parallelism rather than merely halved depth; the collision residual is exchanged once at the end (`MessageType.rsf_collision_residual`), and per-layer gradients stay shard-local until `backwardRSF`-style gather. `RSFMidpointShardResult` mirrors `RSF.MidpointResult` plus `frontier_cores: [2]usize` and `exchanges: usize`.
   - `pub fn stageRSFShardsOnDevices(self: *RelationalGraphProcessingUnit) !void` — when `p2pAvailable()`, uploads each layer's weight tensors to device shards via the existing `P2PTransferManager.stageShardOnDevice` (bytes = s ‖ t per layer, layer-major); without CUDA it returns `error.CudaRuntimeUnavailable` (the existing `initDisabledP2P` path keeps the fabric on-CPU and fully functional).
4. Integration of the existing managers with real RSF signals:
   - `DynamicEdgeWeighting`: per-core edge weights updated from `|Shard.logdet_contribution|` (normalized by the layer mean) — replace any synthetic/random inputs currently used in its update path with this measured signal when invoked through the new RSF paths (existing graph-only update functions remain for the NSIR-only API surface).
   - `SparseActivationManager`: channel masks from shard scale saturation (`pre_s` at the clip bounds → channel inactive for the scatter-flow mask; expose `pub fn shardChannelMask(self, layer: usize, allocator) ![]bool`). Additionally, from the causal path: `pub fn shardTokenActivityMask(self, layer: usize, allocator) ![]bool` marking tokens whose `s_t` saturated — a measured sparsity signal for the sequence flow.
   - `PowerGatingController`: idles cores whose shard gradient L2 norm is below `setSparsityThreshold` after `backwardRSF`; during `midpointCollisionRSF` it may gate a core group only after its frontier completes (never mid-frontier — state this rule in the doc comment); statistics reflect real measured idle cycles.
   - `GraphIsomorphismProcessor`: `processIsomorphismParallel` unchanged; add `pub fn mapLayersToCores(self: *RelationalGraphProcessingUnit, model: *const RSF) !void` placing identical layers (equal weight hashes via the existing tensor hashing) on the same core group — used by `distributeRSFModel` as an optional placement hint before the round-robin default.
   - `distributeGraph`, `distributeGraphFast`, `propagateWeightsAsync`, `synchronizeGraphs`, `managePower`, statistics: preserved for the NSIR graph API, with node payloads now expected to carry latents from `VPU.computeGraphLatents` where graph steps execute coupling ops, and with `propagateWeightsAsync` delegating its signal propagation to the bit-plane kernel of Phase 11; the pure-topology paths remain functional for the C API's graph operations.
5. Tests (under a new `test-rgpu` — register in `build.zig` and add to `test-all`): sharded forward/inverse equivalence gate above on a dim-32/4-layer model with a 2×2 grid and a 4×2 grid, **with diffusion enabled and disabled**; `diffusion_exchanges == 0` when disabled and `> 0` when enabled with `cores > 1`; `forwardRSFSequence`/`inverseRSFSequence` roundtrip `< 5e-7` and equality with `RSF.forwardSequence` within the shard gate; `midpointCollisionRSF` on `L = 8` reproduces `RSF.midpointCollision`'s collision loss within `1e-4` and assigns the two frontiers to disjoint core sets (asserted); `backwardRSF` parameter grads match `model.backwardWithLogDet` within abs 1e-4/rel 1e-3; log-det accumulation matches `model.forwardWithLogDet` within 1e-3; P2P-unavailable environment (the default in CI) still executes all CPU paths; power gating and sparse masks react to real gradient/scale/saturation values (construct a model with a saturated dimension and assert the mask).

## 16. Phase 11 — `src/core_relational/nsir_core.zig`: branchless bit-plane popcount signal propagation

Replace the scalar, per-neighbor, branch-inside-the-loop implementation of `bitmaskSignalPropagate` (line 497) with SIMD bit-plane popcount accumulation, so relational signal propagation is divergence-free and matches the R-GPU's bit-serial hardware model.

1. **Public signatures are preserved exactly**, because `src/core_relational/c_api.zig` imports this module and must compile untouched (Section 3, constraint 10):
   ```zig
   pub fn bitmaskSignalPropagate(bitmask: []const u64, node_count: usize, signal: []const f32, decay: f32, out: []f32) void
   pub fn bitmaskSignalPropagateExact(bitmask: []const u64, node_count: usize, signal: []const f32, decay: f32, out: []f32) void  // new: exact f32/f64 reference and OOM fallback
   ```
   `bitmaskSignalPropagateExact` holds the current semantics (`out[i] = signal[i] + decay · Σ_{j : C[i][j]=1} signal[j]`, accumulated in f64 and cast once) and is the normative reference for tests and the allocation-failure fallback.
2. **Algorithm** (offset-binary 8-bit planes with row-popcount bias correction). The source plan's sketch quantizes as `q = clamp(signal·255/max_abs, 0, 255)`, which maps every negative signal to `0` and therefore silently discards the negative half of the distribution. That is a correctness defect, and the required implementation corrects it while keeping the plan's bit-plane popcount structure:
   ```
   guards (unchanged contract): if node_count == 0 or signal.len < node_count or out.len < node_count -> return
   words = bitmaskWordCount(node_count); if bitmask.len < node_count*words -> return
   max_abs = max_{j < node_count} |signal[j]|
   if max_abs <= 1e-8:  out[i] = signal[i] for all i < node_count; return          // nothing propagates; out is fully written
   quant_scale = 127.5 / max_abs                                                   // offset-binary midpoint
   q[j] = u8( clamp( (signal[j] + max_abs) * quant_scale, 0, 255 ) )               // branchless clamp, q in [0,255]
   planes[p][w] |= bit j   for each j with (q[j] >> p) & 1 == 1, w = j >> 6, bit = j & 63, p in 0..7
   for each i < node_count:
       row  = bitmask[i*words .. (i+1)*words]
       sum_q = Σ_{p=0..7} 2^p · Σ_{w} popCount(row[w] & planes[p][w])              // no data-dependent branch
       k_i   = Σ_{w} popCount(row[w])                                              // neighbour count of row i
       approx_sum = sum_q / quant_scale − max_abs · k_i                            // f64 accumulator, cast once
       out[i] = signal[i] + decay · approx_sum
   ```
   Rationale to record in the doc comment: `signal[j] ≈ q[j]/quant_scale − max_abs`, so summing the dequantized plane counts and subtracting the offset `max_abs` once per neighbour yields the same quantity the exact loop computes, with no per-neighbour branch and no per-neighbour float multiply.
3. **Error bound (normative, asserted by tests).** Each `q[j]` carries rounding error ≤ 0.5, so per neighbour the error is ≤ `0.5/quant_scale = max_abs/255`, and
   `|approx_sum − exact_sum| ≤ k_i · max_abs / 255`, hence
   `|out_simd[i] − out_exact[i]| ≤ |decay| · k_i · max_abs / 255 + 1e-5` (the additive term is f32 accumulation slack).
   State this bound in the doc comment of both functions. The bit-plane path is an intentional, bounded-precision hardware-faithful approximation; it is **not** a substitute for the exact path where exactness is required, and every production caller must be documented as using one or the other deliberately.
4. **Branchlessness and vectorization.** The accumulation loops contain no data-dependent control flow: `inline for (0..7)` is comptime-unrolled, `@popCount` is a single instruction, clamping uses `@min`/`@max`. When `words >= 8`, accumulate the plane popcounts with `@Vector(8, u64)` loads and a vector popcount reduction, with a scalar tail; plane storage is `[8][words]` contiguous so every inner loop is stride-1.
5. **Allocation and failure policy.** The plane buffer is `8·words` `u64`s (`≈ node_count` bytes), allocated from `std.heap.smp_allocator` as in the source plan; on allocation failure the function **falls back to `bitmaskSignalPropagateExact`** — fully functional, slower. It must never return with `out` partially written. For testability, factor the allocating path into `fn propagatePlanes(allocator: Allocator, bitmask, node_count, signal, decay, out) !void` (returning `error.OutOfMemory`) and have both public functions call it, `bitmaskSignalPropagate` catching the error and falling back.
6. **Callers to migrate** (grep-verified list required in the delivery notes): the colocated `nsir_core` test at line 1646, `BitmaskMatrix.signalPropagate` in `src/core_relational/vpu.zig` (Phase 9), `RelationalGraphProcessingUnit.propagateWeightsAsync` and any `DynamicEdgeWeighting` update path in `src/core_relational/r_gpu.zig` (Phase 10), and `exportAdjacencyBitmask` consumers. Each caller states in a comment whether it uses the bounded bit-plane path or the exact path, and why.
7. Tests (under `test-nsir`): equivalence to `bitmaskSignalPropagateExact` within the bound of item 3 for `node_count ∈ {1, 3, 7, 63, 64, 65, 128, 511, 512}` × mask densities `{0, 0.01, 0.5, 1.0}` × signal distributions `{all positive, all negative, mixed, one dominant}` — the mixed-sign cases are the regression guard for the defect corrected in item 2; `decay == 0` ⇒ `out == signal`; all-zero mask ⇒ `out == signal`; all-zero signal ⇒ `out == signal`; Phase 0 fixture (g) reproduced within the bound; `propagatePlanes` under `std.testing.FailingAllocator` returns `error.OutOfMemory` and the public wrapper still produces the exact result (fallback proof); measured timing for `node_count = 4096` at density 0.5 recorded against the Phase 0 baseline (d) — the bit-plane path must not be slower than the exact reference.

## 17. Phase 12 — Dependent migration (bounded, no redesign)

1. `src/distributed/distributed_trainer_futhark.zig`:
   - Replace the host-side per-layer `fisher_s/fisher_t/momentum/master` arrays with one `SFD` instance bound to the trainer's `RSF` model; the CPU optimizer job calls `sfd.step`; hyperparameters (`fisher_gamma`, `fisher_epsilon`, `weight_floor`, `trust_ratio`, warmup) map from `TrainerConfig` into `SFDConfig` at construction.
   - **Checkpoint format**: `src/distributed/checkpoint_envelope.zig` `VERSION` goes `7 → 8`. The v8 RSF optimizer block writes `master_s ‖ master_t ‖ momentum_s ‖ momentum_t ‖ fisher_blocks_s ‖ fisher_blocks_t` via `SFD.exportFlatState` (Fisher arrays now `num_layers·half·3`). Implement a v7 legacy reader that reads the two diagonal Fisher arrays and promotes them with `accel.promoteDiagonalFisher` (`F_ww ← fisher_s`, `F_bb ← fisher_t`, `F_wb ← 0`), and prove it by loading the Phase 0 fixture (e) and asserting the promoted blocks and the resulting first `sfd.step` output match the documented diagonal-equivalence property of Section 4.6 (`B = 0` ⇒ identical step within `1e-6`).
   - **Midpoint training**: `trainPreparedStepFuthark` invokes `accel.fusedMidpointTrainingStep` when `TrainerConfig.midpoint_training` is true (new field, default `true`, env override `JAIDE_MIDPOINT`), which executes the forward projection on layers `[0 … M−1]` and the inverse projection on layers `[M … L−1]` in one kernel launch without writing intermediate layer activations to global device memory. Keep the existing single-frontier `fusedTrainingStep` path intact and selectable (`midpoint_training = false`) so gate 9(b) can be measured from the same binary; record which path ran in every step log line.
   - `StepResult` gains `collision_loss: ?f32 = null`, `logdet_forward: ?f32 = null`, `logdet_backward: ?f32 = null`. They are populated only in midpoint mode; in single-frontier mode they stay `null` and the step log prints `n/a`. Writing `0`, or copying another loss into them, fabricates a number that was never computed and is a defect.
   - `reconstruction_alpha` remains a live hyperparameter of the single-frontier path; in midpoint mode the reconstruction term is subsumed by the collision residual (Section 4.7, Phase 6 item 4) — document this at the config field, and do not silently ignore the field.
   - **Spectral normalization**: the periodic coupling normalization triggered by `spectral_interval` calls `accel.spectralNormalizeLayers(config.spectral_target_norm)` (single exact pass, no `iterations` argument), and the trainer's `sfd.SpectralNormalizer` usage for coupling stacks is replaced by `tensor.constrainCouplingSpectralNorm`. `applyEmbeddingSpectralNormalization` (line 2554) **keeps** its existing iterative dense-embedding normalization with `u`/`v` and `power_iterations` — the source plan's instruction to use the exact rank-2 formula applies to coupling matrices, and the embedding operand is `[vocab_size][dim]`, for which the 2×2 Gram closed form would return a mathematically wrong `σ_max` (Section 4.9 scope rule). Add the scope comment at that call site.
   - **Causal sequence training** (optional path, off by default): `TrainerConfig.causal_sequence_coupling: bool = false`, env `JAIDE_CAUSAL`. When enabled, the trainer builds the mask with `VPU.relationalCausalMask` if a relational graph is loaded for the batch, otherwise with `VPU.causalMaskFromSequence`, and routes the device step through `accel.causalSequenceForward`/`causalSequenceInverse` per layer around the coupling kernels. Log the measured mask density (`VPU.causalMaskDensity`) per step.
   - GPU memory admission: `EstimateConfig` gains `.fisher_block = true`; the admission failure messages keep their existing names.
2. `src/main_distributed_futhark.zig`: keep `JAIDE_MODEL_DIM` (default 98304) and `JAIDE_LAYERS` (default 24); add `JAIDE_DIFFUSION` (default `1`), `JAIDE_MIDPOINT` (default `1`), `JAIDE_CAUSAL` (default `0`), each validated with the existing `resolveEnvUsizeDefault`-style error reporting. At startup print the resolved geometry: `model_dim`, `half`, `num_layers`, and the diffusion layout from `types.rsfDiffusionLayout(model_dim)` as `radix/block/stages` (for the default: `radix=3 block=32768 stages=15`), plus `midpoint_frontier_layers = num_layers / 2`.
3. `src/core/model_io.zig`: `CURRENT_VERSION` 1→2 with a v1 legacy reader (v1 sections parse exactly as today; v2 adds SSI latent payloads with the layer-map identity, the Ranker head plus head-SFD block state, and RSF v7 model files). `ModelMetadata` gains `rsf_save_version: u32`, `has_latent_index: bool`, `global_diffusion: bool`, `fisher_layout: u8` (2 = legacy diagonal, 3 = block). `saveSFDState`/`loadSFDState` re-export the new SFD API. Round-trip test: export v2, import v2, compare tensors; import the v1 golden fixture.
4. `src/api/inference_server.zig`: the long-context pipeline flows embedded inputs through `OFTB` directly; the causal bitmask path becomes the default when `sequence_training` is enabled, and the single-frontier midpoint path stays available behind the existing `reconstruction_alpha` configuration.
5. `src/semantic_check_root.zig`, `src/_bench_deps.zig`, `src/test_root_*.zig`: update imports/uses to the new APIs; test roots remain thin wrappers. Add `src/test_root_rgpu.zig` (Phase 10) and `src/test_root_rsf_native.zig` (Phase 13), both registered in `build.zig` `test_specs` and `test_all_step`. `src/_bench_deps.zig` re-exports `pub const sfd = @import("optimizer/sfd.zig")` and `src/tests/bench_sfd.zig` uses `deps.sfd.Tensor`; that symbol is deleted in Phase 4, so the bench must switch to `deps.tensor.Tensor` (or the `jaide` module) and its private FP4 replication must be removed with it.
6. Benchmarks (`src/tests/bench_*.zig`, registered through the `deps` module): add cases `couplingForwardBatch`/`couplingInverseBatch`/`diffuseRowInPlace` (row_len 98304)/`causalCouplingForwardBatch` to `bench_tensor_ops.zig`; `midpoint_vs_sequential` (gate 9(b): `L = 24`, `dim = 512`, batch 4, seq 64, ReleaseFast, median of 20) and `spectral_exact_vs_iterative` (gate 10: one `[49152, 2]` stack, exact vs the recorded 30-iteration baseline) to `bench_rsf.zig`; the model-bound SFD block step, the FP16 writeback, and the dense spectral normalizer to `bench_sfd.zig` (remove the private FP4 replication — `quantizeValue` is deleted with the private tensor stack — and replace those sections with the kernel benchmarks above); `bitplane_vs_exact_propagate` (node_count 4096) to a new or existing relational bench.
7. `src/hw/rtl/rtl_sim_main.zig` and the Clash sources `RankerCore.hs`, `SSISearch.hs`: update the Haskell models to the new fused-score arithmetic in fixed point (`Fix16` dot products over quantized latents, SimHash bit agreement count, the four-criteria weighted sum with the same constants) and update `rtl_sim_main.zig` to the new SSI API via the `jaide` module. If any of these sources models spectral normalization iteratively, replace it with the closed-form 2×2 Gram expression of Section 4.9; if none does, state that in the delivery notes rather than inventing a change. `-Drtl` requires GHC/Clash: attempt the build, and if the toolchain is absent, deliver the updated sources and report the path as unverified — do not claim it passed.
8. `src/core_relational/mod.zig` and `c_api.zig`: no C ABI changes; `mod.zig` re-exports unchanged names (add the new `types.RSFSequenceMask`/`bitmaskSignalPropagateExact` re-exports only if `mod.zig` already re-exports the corresponding old names); `c_api.zig` must compile untouched (it depends only on `nsir_core`/ESSO — verify by building `c-api` before and after). `src/core_relational/formal_verification.zig` (2894 lines) imports `nsir_core.SelfSimilarRelationalGraph`, `Node`, `Edge`, `EdgeQuality`, and `EdgeKey` and is exercised by `test-formal` (27 tests, verified passing today); the Phase 11 rewrite must keep those types and their public signatures intact so this module compiles unchanged, and `test-formal` is part of gate 2.

9. **`src/core_relational/dataset_obfuscation.zig`**: append `GlobalDiffusion`, `CausalBitmask`, `SignalPropagation`, `LayerMidpoint`, `NaturalGradientBlock`, and `SpectralNormRank2` to the existing `OperationType` enum (`Matmul`, `Softmax`, `LayerNorm`, `Residual`, `Gelu`, `EmbeddingLookup`). That enum labels every traced operation, so leaving it unchanged silently mislabels all six native operations as generic matrix products in the obfuscated dataset and in any trace consumer. This is a six-line change plus a test that every enum value round-trips through obfuscation.

## 18. Phase 13 — Cross-module invariant test suite (`test-rsf-native`)

Create `src/test_root_rsf_native.zig` importing all six modules plus `rsf.zig`, `oftb.zig`, `types.zig`, and `nsir_core.zig`, containing at minimum:

1. `rsfNative_roundtripAllModules`: one dim-16/3-layer model; a latent flowed via `RSFLatentState.forwardThrough`, reconstructed via a VPU lane path, executed via RGPU `forwardRSF` on a 2×2 grid, and inverted back — all three execution paths agree within their gates and `inverse∘forward` recovers the input to ≤ 1e-4 relative, with diffusion enabled and disabled.
2. `rsfNative_logDetConsistency`: the sum of per-layer `forwardLatentWithLogDet` log-dets equals the whole-stack log-det within 1e-4, equals a finite-difference log-det of the stack map on dim ≤ 4 within rel 1e-2, and is unchanged by the orthogonal factors (enable/disable diffusion and assert the log-det is identical to within 1e-6 — `Θ` and `R` contribute exactly 0).
3. `rsfNative_bindingEnforcement`: every public API of the six modules rejects unbound/misbound tensors with the typed `RSFBindingError` (enumerate the attempted misuses in the test).
4. `rsfNative_sfdConvergence`: the SFD convergence scenario of Phase 4 executed end-to-end through the trainer's CPU optimizer job migrated in Phase 12, asserting gate 11 (steps to 50% of initial loss ≤ baseline/3 from `docs/upgrade/sfd-convergence.txt`) and invertibility maintained.
5. `rsfNative_ssiRankerPipeline`: embed → forward → index → retrieve → rank on a fixed corpus; assert the top-1 segment for each query is the planted nearest neighbour for ≥ 9/10 queries; assert `RSFModelMismatch` when the query latent was produced with a different `global_diffusion` setting.
6. `rsfNative_goldenCompatibility`: load every Phase 0 fixture (a)–(h) with the new code and match recorded outputs within the fixtures' tolerances.
7. `rsfNative_noStandalonePaths`: compile-time behavioral check — construct each module only through RSF handles; assert the removed generic entry points no longer exist by calling the module's current public API surface (this test documents the enforced surface and fails to compile if the generic forms reappear).
8. `rsfNative_diffusionGlobalMixing`: `Θ` on the production geometry (`row_len = 98304`, `radix = 3`, `block = 32768`, `stages = 15`) and on `row_len = 32` (`radix = 1`, `stages = 5`): involution `‖Θ(Θ(x)) − x‖_∞ < 1e-7`, energy `|‖Θx‖₂ − ‖x‖₂| < 1e-7`, a single-coordinate perturbation producing non-zero deltas in all `radix` blocks, `Θ` equal to its own transpose (compare `⟨Θa, b⟩` with `⟨a, Θb⟩` within 1e-6), and measured stage count equal to `stages` (instrumented counter).
9. `rsfNative_exactSpectralEquivalence`: `tensor.exactSpectralNormRank2`, `rsf.exactSpectralNormRank2`, and (when the Futhark toolchain is available) `stack_spectral_normalize_exact` all agree with an f64 Jacobi eigenvalue reference and an angular-sweep reference within `2.0e-6` on 1,000 PRNG coupling matrices; assert the coupling normalization API accepts no iteration count (call it with `(weight, target)` only — the test fails to compile if the parameter returns).
10. `rsfNative_blockNaturalGradient`: `‖M^{-1/2} M M^{-1/2} − I‖₂ < 1e-5` over 10,000 PRNG SPD blocks; `B = 0` degeneracy equals the diagonal step within `1e-6`; `F_wb` has the sign of the gradient correlation after one step; `condition_number_max` in `StepStats` matches an independent f64 eigenvalue computation within `1e-6` relative.
11. `rsfNative_midpointEquivalence`: with `y* = F(x₀)` from the ordinary forward on `L = 24`, `dim = 64`, `midpointCollision` returns `collision_loss < 1e-6`, `forward_layers + backward_layers == L`, the layer-application counter for the two frontiers is `≤ 2·⌈L/2⌉`, and `midpointBackward` gradients match single-frontier `backwardWithLogDet` gradients within rel `1e-3`; `logdet_total` equals `forwardWithLogDet`'s log-det within `1e-4`.
12. `rsfNative_causalCrossTokenFlow`: roundtrip `max(‖X1−X1_rec‖_∞, ‖X2−X2_rec‖_∞) < 5.0e-7` on seq ≤ 64 / dim ≤ 32; strict causality (mutating token `t'` leaves outputs at `t < t'` bit-identical); `logdet = Σ_t Σ_d s_{t,d}` within 1e-5; all-zero mask reproduces the per-token path exactly (fixture (h)); a relational mask from `VPU.relationalCausalMask` on a real `SelfSimilarRelationalGraph` differs from the full causal mask exactly on the non-adjacent token pairs; a non-lower-triangular mask is rejected with `error.InvalidCausalMask`.
13. `rsfNative_bitplanePropagation`: `nsir_core.bitmaskSignalPropagate` versus `bitmaskSignalPropagateExact` within `|decay|·k_i·max_abs/255 + 1e-5` for mixed-sign signals; failing-allocator fallback produces the exact result; the C API's exported symbol list is byte-identical to the Phase 0 record.
14. `rsfNative_invariantParadigm`: enumerate every learnable tensor in a fully populated system (RSF model layers, SFD masters/momentum/Fisher blocks, ranker head, head SFD) and assert each learnable parameter tensor has space `.layer_weight_s`, `.layer_weight_t`, or `.ranker_head` and shape `[dim, 2]` — i.e. no dense learned projection matrix exists anywhere; assert no softmax/attention/convolution/recurrent symbol is reachable from the six modules (the grep gates of Phase 14 are the static counterpart of this test).
15. `rsfNative_liouvilleDepthScaling`: for `L ∈ {8, 24, 64}` on a fixed input distribution and spectrally constrained weights, assert `roundtripError ≤ 1e-4`, `resonanceDrift` finite and reported per 8-layer boundary, and the log-det equal to the sum of per-layer contributions within 1e-3 — with no normalization, residual, or dropout component present in the layer map. Prove the layer map is exactly `Θ ∘ R ∘ C` constructively: zero every coupling weight (`s_w = s_b = t_w = t_b = 0` ⇒ `C = identity`, `logdet = 0`) and disable diffusion, so the `L`-layer stack must equal `R^L`; assert that for `L = 8` the stack is the identity within `1e-6` (i.e. `R^8 = I`, Section 4.4) and for `L = 24` it is the identity as well, and that any additive/multiplicative per-layer term not in `Θ ∘ R ∘ C` would break this equality. Also assert finiteness and report conditioning instead of assuming it: for each `L`, take `RSF.stateGrowthReport` (Phase 3) and assert every reported `max|x|` is finite and below `65504`, so the same weights stay representable on the FP16 device path; if some `L` overflows, report the measured depth limit. Raising the clip bounds or inserting normalization to make the test pass violates Section 3, constraint 4 and is a gate failure.
16. Registration in `build.zig` follows the Phase 0 per-spec accelerator policy and splits the suite in two, so that the invariants stay executable in an environment without Futhark. `test-rsf-native` (wrapper `src/test_root_rsf_native.zig`, registered **without** `applyAccel`) holds every test whose import closure does not reach `src/processor/rsf.zig` or `src/hw/accel/` — the tensor/OFTB/SFD/NSIR-level invariants (exact spectral norm on `tensor.zig`, block natural gradient, global diffusion involution and mixing, bit-plane propagation versus its exact reference, binding enforcement for the modules that do not import the accelerator, the invariant-paradigm enumeration, causal coupling at the OFTB/tensor level). `test-rsf-native-accel` (wrapper `src/test_root_rsf_native_accel.zig`, registered **with** `applyAccel`) holds everything that imports `rsf.zig`, the migrated trainer, the RSF model fixtures, the midpoint frontier path, or the device kernels. Both specs are added to `test-all`. Verified reason for the split: `src/processor/rsf.zig` imports `src/hw/accel/accel_interface.zig`, whose container-level `@import("build_options")` and whose `pub extern "c" fn futhark_*` declarations make any artifact that reaches it fail without the build-provided options module and the generated `main_cpu.c` (undefined symbols `futhark_new_u64_1d`, `futhark_values_i64_1d`, `futhark_project_opaque_tup7_…` at link time). When Futhark is unavailable, report `test-rsf-native` green and `test-rsf-native-accel` blocked with that link output attached; never collapse the two into one spec that is uniformly blocked.

## 19. Phase 14 — Final verification, performance, and delivery

1. Run the full gate list from Section 1 and record outputs: `zig build check`, `zig build test-all`, `zig build c-api`, `zig build test-c-api`, `zig build -Doptimize=ReleaseFast bench`, `zig build -Doptimize=ReleaseFast inference-server`, `zig build -Doptimize=ReleaseFast distributed-server`, `zig build -Dcpu=native -Daccel=cuda test-hybrid` if that spec exists in the final `build.zig` (skipped with the reason recorded when no CUDA device is present), and the direct sweep `for w in src/test_root_*.zig; do zig test "$w"; done`. For each gate record pass / fail / blocked and the per-spec test counts. For every Futhark-dependent artifact additionally record `futhark --version` output, whether `zig build regen-futhark` produced `src/hw/accel/{main_cpu,main_gpu,futhark_kernels}.{c,h}`, and — for `test-rsf` — the exact link-time undefined-symbol text if it is still blocked. The report separates the three classes explicitly: gates executed and passed, gates executed and failed, gates blocked by the missing `futhark` binary. Nothing blocked is reported as passed, and no blocked measurement is substituted by an estimate.
2. Quantitative acceptance criteria (all must be measured and reported with the command that produced them):
   - `exactSpectralNormRank2` matches double-precision SVD/Jacobi singular values within `|σ_exact − σ_svd| < 2.0e-6`.
   - The analytic 2×2 block inverse square root satisfies `‖(F_d + λI)^{-1/2}(F_d + λI)(F_d + λI)^{-1/2} − I‖₂ < 1e-5`.
   - The Fast Walsh-Hadamard / `Q_r ⊗ H_{2^k}` diffusion satisfies exact involution `Θ(Θ(x)) = x` and energy preservation `‖Θ(x)‖₂ = ‖x‖₂` within `1e-7`.
   - Causal bitmask coupling inversion error satisfies `max(‖X1 − X1_rec‖_∞, ‖X2 − X2_rec‖_∞) < 5.0e-7`.
   - Midpoint path per-step wall clock ≤ `0.6×` the single-frontier per-step wall clock on the gate 9(b) workload, with per-frontier sequential layer applications `⌈L/2⌉`.
   - SFD steps-to-target ≤ `1/3` of the Phase 0 diagonal baseline, reported **together with** the measured mean `|ρ|` and the `(1+|ρ|)/(1−|ρ|)` bound of Section 4.6. If the ratio is not met, report the measured ratio for both Fisher modes plus the `ρ` statistics and mark the numeric target as not met on this workload (gate 11's honesty clause); the delivery is not blocked by this single number, but the number must never be massaged.
   - Bit-plane propagation within the bound of Section 16 item 3 and not slower than the exact reference.
   - `zig build check` free of semantic, type, and alignment errors; `zig build test-all` with zero test failures.
3. Grep gates — all must return zero matches:
   - `grep -rn "fromCoreTensor\|toCoreTensor" src/`
   - `grep -n "pub const Tensor = struct" src/optimizer/sfd.zig`; `grep -n "param_size" src/optimizer/sfd.zig`
   - `grep -n "vectorScore\|dotProductScore" src/ranker/ranker.zig`
   - `grep -rn "TODO\|FIXME\|unimplemented\|notImplemented" src/core/tensor.zig src/optimizer/sfd.zig src/index/ssi.zig src/ranker/ranker.zig src/core_relational/vpu.zig src/core_relational/r_gpu.zig src/core_relational/nsir_core.zig src/processor/rsf.zig src/processor/oftb.zig src/hw/accel/main.fut src/hw/accel/futhark_kernels.fut src/hw/accel/accel_interface.zig src/distributed/distributed_trainer_futhark.zig`
   - `grep -rn "LAYER_SPECTRAL_POWER_ITERATIONS\|spectralNormPowerIteration" src/` → zero matches.
   - `grep -n "power_iters" src/hw/accel/main.fut` and `grep -rn "power_iterations" src/optimizer/sfd.zig src/hw/accel/accel_interface.zig src/distributed/distributed_trainer_futhark.zig` → every surviving match must lie inside the **dense embedding** normalization path (`entry embedding_spectral_normalize`, `spectralNormalizeEmbedding`, `SpectralNormalizer`'s dense-operand branch, `applyEmbeddingSpectralNormalization`); list each match with file, line number, and its Section 4.9 justification in the delivery notes. Any match on a coupling path is a gate failure.
   - `grep -rn "powerIteration" src/core_relational/vpu.zig src/core_relational/r_gpu.zig src/index src/ranker` → any surviving match must be a diagnostic on arbitrary dense operands (Phase 9 item 3) and must not be reachable from a coupling normalization call; list them as above.
   - `grep -rn "constrainCouplingSpectralNorm\|constrainCouplingStackSpectralNorm\|couplingSpectralNorm\|exactSpectralNormRank2\|stack_spectral_normalize_exact" src/` → enumerate every coupling normalization call site in the delivery notes; each one must be a single exact pass with no iteration-count argument.
   - `grep -rn "softmax\|attention\|query_key\|qk_t\|conv2d\|convolution\|lstm\|gru_cell\|recurrent" src/processor src/optimizer src/core/tensor.zig src/core/types.zig src/index src/ranker src/core_relational`
   - `grep -rn "stack_update_sfd_master\|futhark_entry_stack_spectral_normalize\b" src/`
4. Static confirmation of RSF-only construction: in each of the six modules, every `pub fn init*` takes `*const RSF`, `*RSF`, or a bound-tensor/latent/mask argument as its provenance source; grep-verify and list the constructors in the delivery notes.
5. Delivery format: for every created or modified file, provide the complete final file content with its full repository path, unabbreviated. Never emit diffs, excerpts, `// ... rest unchanged`, elided bodies, or TODO markers. Files to deliver in full: `src/core/tensor.zig`, `src/core/types.zig`, `src/core/memory.zig`, `src/core/coupling.zig`, `src/core/sfd.zig`, `src/core/ranker.zig`, `src/core/model_io.zig`, `src/processor/oftb.zig`, `src/processor/vpu.zig`, `src/processor/rgpu.zig`, `src/processor/ssi.zig`, `src/hw/accel/main.fut`, `src/hw/accel/futhark_kernels.fut`, `src/hw/accel/futhark_bindings.zig`, `src/distributed/distributed_trainer_futhark.zig`, `src/core_relational/nsir_core.zig`, `src/core_relational/dataset_obfuscation.zig`, `src/main_distributed_futhark.zig`, `src/tests/test_rsf_native_root.zig`, `build.zig`, `docs/rsf_native_baseline.txt`, and `src/api/inference_server.zig` (whose changed regions are re-emitted as complete functions, since the file is 7081 lines).
   - (a) the exact commands run and their exit status/output tails;
   - (b) the before/after benchmark table (Phase 0 baseline vs final medians, including the gate 9(b), 10, 11, and 16-item-3 measurements);
   - (c) a compatibility matrix (which legacy formats load — RSF v6, SFD v1 `0x53464433`, checkpoint v7, model_io v1, SSI legacy, ranker legacy — each with its fixture proof);
   - (d) a list of removed public symbols and their replacements (including `stack_spectral_normalize`, `stack_update_sfd_master`, `spectralNormalizeLayers`'s `iterations` parameter, `SFD.init(allocator, param_size)`, `update`, `applySlices`, `vectorScore`, `dotProductScore`, `calibrateWeights`, `updateWeights`, `computeGraphEmbeddings`);
   - (e) any environment limitation that prevented a gated verification (GPU, Futhark, CUDA, GHC/Clash) — stated as fact, never worked around with fake results;
   - (f) the recorded numbers for gates 9–15, each with the command and the file it is stored in;
   - (g) a theory-to-code mapping table: for each claim in Section 4.4 (tame diffeomorphic integrator, `Aut(AD)` automorphism, Liouville information preservation, `R^8 = I` C8 resonance, `Θ² = I` diffusion involution, exact additive log-det, Three-Mark/Euler–Arnold–Liouville action), the implementing function(s) and the test(s) that verify it.

## 20. Work order summary

Execute Phase 0 through Phase 14 in order: 0 baseline and build graph → 1 `tensor.zig` (coupling algebra, exact spectral norm, diffusion, causal kernels) → 2 `oftb.zig` (`R` + `Θ`) → 3 `rsf.zig` (delegation, exact spectral norm, latent states, causal flow, midpoint frontiers, save v7) → 4 `sfd.zig` (RSF-only, analytic 2×2 block natural gradient) → 5 Futhark kernels → 6 Futhark bindings and `accel_interface.zig`/`gpu_memory.zig` → 7 `ssi.zig` → 8 `ranker.zig` → 9 `vpu.zig` → 10 `r_gpu.zig` → 11 `nsir_core.zig` bit-plane propagation → 12 dependent migration (trainer, checkpoints, model_io, inference server, RTL, benches) → 13 `test-rsf-native` → 14 final gates and delivery.

At the end of every phase, run `zig build check`, `zig build c-api`, `zig build test-c-api`, and the direct sweep `for w in src/test_root_*.zig; do zig test "$w"; done`; once the Phase 0 build-graph correction is in place and if the `futhark` binary is available, also run `zig build test-all` (plus `zig build regen-futhark` first, since the accelerator C sources are gitignored). Fix every regression before starting the next phase: the 562 tests verified passing today and the 180 C-API tests are a floor, not a target. The upgrade is complete only when every gate in Sections 1 and 19 either passes on the real toolchain or is reported as blocked-on-Futhark with its exact failing output, all delivered files are complete and unabridged, and the six modules plus the substrate contain no standalone, RSF-agnostic code path.
