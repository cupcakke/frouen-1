entry matmul [m][n][k] (a: [m][k]f32) (b: [k][n]f32): *[m][n]f32 =
  let bt = transpose b
  in map (\a_row -> map (\b_col -> f32.sum (map2 (*) a_row b_col)) bt) a

let oftb_scale_f32 : f32 = 0.7071067811865476

let sanitize_f32 (v: f32) : f32 =
  if f32.isnan v || f32.isinf v then 0f32 else v

let clamp_f16_value (v: f32) : f32 =
  let safe = sanitize_f32 v
  in f32.max (-60000f32) (f32.min 60000f32 safe)

let clamp_f16_weight (v: f32) : f32 =
  let safe = sanitize_f32 v
  in f32.max (-65504f32) (f32.min 65504f32 safe)

let splitmix64 (value: u64) : u64 =
  let z0 = value + 0x9E3779B97F4A7C15u64
  let z1 = (z0 ^ (z0 >> 30u64)) * 0xBF58476D1CE4E5B9u64
  let z2 = (z1 ^ (z1 >> 27u64)) * 0x94D049BB133111EBu64
  in z2 ^ (z2 >> 31u64)

let rsf_stack_coupling_row [half]
  (row: [half*2]f32)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min_f32: f32) (clip_max_f32: f32) : [half*2]f32 =
  let x1 = row[0:half] :> [half]f32
  let x2 = row[half:half*2] :> [half]f32
  let scale = map (\d ->
    let sum = f32.f16 weights_s[d][0] * x2[d] + f32.f16 weights_s[d][1]
    let clipped = f32.max clip_min_f32 (f32.min clip_max_f32 sum)
    in f32.exp clipped) (iota half)
  let y1 = map2 (*) x1 scale
  let y2 = map2 (\x2_j j ->
    let trans = f32.f16 weights_t[j][0] * y1[j] + f32.f16 weights_t[j][1]
    in x2_j + sanitize_f32 trans) x2 (iota half)
  let o1 = map2 (\a b -> (a - b) * oftb_scale_f32) y1 y2
  let o2 = map2 (\a b -> (a + b) * oftb_scale_f32) y1 y2
  in (map clamp_f16_value (o1 ++ o2)) :> [half*2]f32

let rsf_stack_invert_row [half]
  (row: [half*2]f32)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min_f32: f32) (clip_max_f32: f32) : [half*2]f32 =
  let y1p = row[0:half] :> [half]f32
  let y2p = row[half:half*2] :> [half]f32
  let u1 = map2 (\a b -> (a + b) * oftb_scale_f32) y1p y2p
  let u2 = map2 (\a b -> (b - a) * oftb_scale_f32) y1p y2p
  let x2 = map (\d ->
    let trans = f32.f16 weights_t[d][0] * u1[d] + f32.f16 weights_t[d][1]
    let safe_trans = sanitize_f32 trans
    in u2[d] - safe_trans) (iota half)
  let x1 = map (\d ->
    let pre = f32.f16 weights_s[d][0] * x2[d] + f32.f16 weights_s[d][1]
    let clipped = f32.max clip_min_f32 (f32.min clip_max_f32 pre)
    in u1[d] / f32.exp clipped) (iota half)
  in (map clamp_f16_value (x1 ++ x2)) :> [half*2]f32

entry rsf_forward [n][half] (input: [n][half*2]f16)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min: f16) (clip_max: f16) : *[n][half*2]f16 =
  let raw_min = f32.f16 clip_min
  let raw_max = f32.f16 clip_max
  let clip_min_f32 = f32.min raw_min raw_max
  let clip_max_f32 = f32.max raw_min raw_max
  in map (\row ->
    let row_f32 = map f32.f16 row
    let result = rsf_stack_coupling_row row_f32 weights_s weights_t clip_min_f32 clip_max_f32
    in map (\v -> f16.f32 (clamp_f16_value v)) result) input

let sfd_fisher_update_core [d][e]
  (weights: [d][e]f32) (gradients: [d][e]f32)
  (momentum_state: [d][e]f32) (fisher_state: [d][e]f32)
  (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32)
  (optimizer_step: i64) (epsilon: f32) (trust_ratio: f32) (weight_floor: f32)
  : ([d][e]f32, [d][e]f32, [d][e]f32) =
  let safe_beta = f32.max 0f32 (f32.min 0.99999994f32 (sanitize_f32 momentum_beta))
  let safe_gamma = f32.max 0f32 (f32.min 0.99999994f32 (sanitize_f32 fisher_gamma))
  let safe_eps = f32.max (sanitize_f32 epsilon) 1e-12f32
  let safe_lr = sanitize_f32 learning_rate
  let safe_trust_ratio = f32.max 0f32 (f32.min 1f32 (sanitize_f32 trust_ratio))
  let safe_floor = f32.max 0f32 (sanitize_f32 weight_floor)
  let step_f = f32.i64 (i64.max 1 optimizer_step)
  let momentum_correction = f32.max safe_eps (1f32 - safe_beta f32.** step_f)
  let fisher_correction = f32.max safe_eps (1f32 - safe_gamma f32.** step_f)
  let momentum_next = map2 (map2 (\m g ->
    let safe_m = sanitize_f32 m
    let safe_g = sanitize_f32 g
    let candidate = safe_beta * safe_m + (1f32 - safe_beta) * safe_g
    in sanitize_f32 candidate)) momentum_state gradients
  let fisher_next = map2 (map2 (\f g ->
    let safe_f = f32.max 0f32 (f32.min 1e6f32 (sanitize_f32 f))
    let safe_g = sanitize_f32 g
    let candidate = safe_gamma * safe_f + (1f32 - safe_gamma) * safe_g * safe_g
    in f32.min 1e6f32 (f32.max 0f32 (sanitize_f32 candidate)))) fisher_state gradients
  let weights_next = map3 (map3 (\w m f ->
    let safe_w = sanitize_f32 w
    let m_hat = m / momentum_correction
    let f_hat = f / fisher_correction
    let raw_step = safe_lr * m_hat / (f32.sqrt (f32.max f_hat 0f32) + safe_eps)
    let max_step = safe_trust_ratio * f32.max safe_floor (f32.abs safe_w)
    let clipped_step = f32.max (-max_step) (f32.min max_step (sanitize_f32 raw_step))
    let updated = safe_w - clipped_step
    in sanitize_f32 updated)) weights momentum_next fisher_next
  in (weights_next, momentum_next, fisher_next)

entry master_weights_to_f16_3d [layers][rows][columns] (weights: [layers][rows][columns]f32): *[layers][rows][columns]f16 =
  map (map (map (\value -> f16.f32 (clamp_f16_weight value)))) weights

entry stack_update_sfd_master [layers][rows][columns]
  (master_weights: *[layers][rows][columns]f32)
  (gradients: [layers][rows][columns]f32)
  (momentum_state: *[layers][rows][columns]f32)
  (fisher_state: *[layers][rows][columns]f32)
  (learning_rate: f32)
  (momentum_beta: f32)
  (fisher_gamma: f32)
  (optimizer_step: i64)
  (epsilon: f32)
  (trust_ratio: f32)
  (weight_floor: f32)
  : (*[layers][rows][columns]f32, *[layers][rows][columns]f32, *[layers][rows][columns]f32) =
  let updates = map4 (\w g m f ->
    sfd_fisher_update_core w g m f learning_rate momentum_beta fisher_gamma optimizer_step epsilon trust_ratio weight_floor
    ) master_weights gradients momentum_state fisher_state
  in (map (\(w, _, _) -> w) updates,
      map (\(_, m, _) -> m) updates,
      map (\(_, _, f) -> f) updates)

entry master_weights_to_f16_2d [rows][columns] (weights: [rows][columns]f32): *[rows][columns]f16 =
  map (map (\value -> f16.f32 (clamp_f16_weight value))) weights

entry embedding_update_sfd_master [vocab_size][dim]
  (master_weight: *[vocab_size][dim]f32) (grad_weight: [vocab_size][dim]f32)
  (momentum_state: *[vocab_size][dim]f32) (fisher_state: *[vocab_size][dim]f32)
  (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32) (optimizer_step: i64) (epsilon: f32)
  (trust_ratio: f32) (weight_floor: f32)
  : ([vocab_size][dim]f32, [vocab_size][dim]f32, [vocab_size][dim]f32) =
  sfd_fisher_update_core master_weight grad_weight momentum_state fisher_state learning_rate momentum_beta fisher_gamma optimizer_step epsilon trust_ratio weight_floor

entry scale_matrix_f32 [rows][columns] (values: *[rows][columns]f32) (scale_factor: f32) : *[rows][columns]f32 =
  map (map (\value -> value * scale_factor)) values

entry clip_matrix_global_norm_f32 [rows][columns]
  (values: *[rows][columns]f32) (clip_norm: f32) : *[rows][columns]f32 =
  let flat_values = flatten values
  let maximum_absolute_value = reduce f32.max 0f32 (map f32.abs flat_values)
  let scaled_norm_squared =
    if maximum_absolute_value > 0f32
    then f32.sum (map (\value ->
      let scaled = value / maximum_absolute_value
      in scaled * scaled) flat_values)
    else 0f32
  let norm = maximum_absolute_value * f32.sqrt scaled_norm_squared
  let scale =
    if clip_norm > 0f32 && norm > clip_norm && norm > 1e-12f32
    then clip_norm / norm
    else 1f32
  in map (map (* scale)) values

entry embedding_forward_padded [n][batch_size][seq_len][vocab_size][dim]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (positions: [seq_len]i64)
  (weight: [vocab_size][dim]f16) : *[batch_size][seq_len][dim]f16 =
  let slot_indices = map2 (\position slot_index ->
    if position >= 0 && position < seq_len then position else slot_index)
    positions (iota seq_len)
  in map2 (\batch_index length ->
    map (\sequence_index ->
      let flat_index = batch_index * seq_len + sequence_index
      in if sequence_index >= 0 &&
            sequence_index < i64.max 0 (i64.min seq_len length) &&
            flat_index >= 0 &&
            flat_index < n
         then let token = tokens[flat_index]
              in if token >= 0 && token < vocab_size
                 then weight[token]
                 else replicate dim (f16.i32 0)
         else replicate dim (f16.i32 0)) slot_indices) (iota batch_size) lengths

entry embedding_backward_padded [n][batch_size][seq_len][dim][vocab_size]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (grad_output: [batch_size][seq_len][dim]f16)
  (grad_weight: [vocab_size][dim]f32) : *[vocab_size][dim]f32 =
  let total = batch_size * seq_len
  let limits = map (\length -> i64.max 0 (i64.min seq_len length)) lengths
  let validity = tabulate total (\flat_index ->
    let batch_index = flat_index / seq_len
    let sequence_index = flat_index % seq_len
    in if sequence_index < limits[batch_index] && flat_index < n
       then let token = tokens[flat_index]
            in token >= 0 && token < vocab_size
       else false)
  let safe_tokens = tabulate total (\flat_index ->
    if validity[flat_index] then tokens[flat_index] else -1i64)
  let masked_grads = tabulate total (\flat_index ->
    if validity[flat_index]
    then let batch_index = flat_index / seq_len
         let sequence_index = flat_index % seq_len
         in map (\v -> sanitize_f32 (f32.f16 v)) grad_output[batch_index][sequence_index]
    else replicate dim 0f32)
  let updates = hist (map2 (+)) (replicate dim 0f32) vocab_size safe_tokens masked_grads
  in map2 (map2 (+)) grad_weight updates

let spectral_normalize_matrix [rows][columns]
  (weight: [rows][columns]f32)
  (target: f32)
  (power_iters: i64)
  : ([rows][columns]f32, f32, f32) =
  let safe_weight = map (map sanitize_f32) weight
  let initial_value = if columns > 0 then 1f32 / f32.sqrt (f32.i64 columns) else 0f32
  let initial_v = replicate columns initial_value
  let initial_u_value = if rows > 0 then 1f32 / f32.sqrt (f32.i64 rows) else 0f32
  let initial_u = replicate rows initial_u_value
  let weight_t = transpose safe_weight
  let (final_u, final_v) =
    loop (u, v) = (initial_u, initial_v) for iteration < i64.max 1 power_iters do
      let _ = iteration
      let _ = u
      let raw_u = map (\row -> f32.sum (map2 (*) row v)) safe_weight
      let u_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_u))
      let safe_u_norm = f32.max u_norm 1e-12f32
      let next_u = map (/ safe_u_norm) raw_u
      let raw_v = map (\column -> f32.sum (map2 (*) column next_u)) weight_t
      let v_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_v))
      let safe_v_norm = f32.max v_norm 1e-12f32
      let next_v = map (/ safe_v_norm) raw_v
      in (next_u, next_v)
  let projected = map (\row -> f32.sum (map2 (*) row final_v)) safe_weight
  let sigma = sanitize_f32 (f32.abs (f32.sum (map2 (*) final_u projected)))
  let safe_target = f32.max target 1e-6f32
  let scale = if sigma > safe_target then safe_target / sigma else 1f32
  let normalized = map (map (* scale)) safe_weight
  in (normalized, sigma, sigma * scale)

entry stack_spectral_normalize [layers][rows][columns]
  (weights: *[layers][rows][columns]f32)
  (target: f32)
  (power_iters: i64)
  : (*[layers][rows][columns]f32, f32, f32) =
  let results = map (\weight -> spectral_normalize_matrix weight target power_iters) weights
  let normalized = map (\(weight, _, _) -> weight) results
  let before = reduce f32.max 0f32 (map (\(_, sigma, _) -> sigma) results)
  let after = reduce f32.max 0f32 (map (\(_, _, sigma) -> sigma) results)
  in (normalized, before, after)

entry embedding_spectral_normalize [vocab_size][dim]
  (weight: *[vocab_size][dim]f32)
  (u: *[vocab_size]f32)
  (v: *[dim]f32)
  (power_iters: i64)
  (target: f32) : (*[vocab_size][dim]f32, *[vocab_size]f32, *[dim]f32, f32, f32) =
  let safe_weight = map (map sanitize_f32) weight
  let weight_t = transpose safe_weight
  let sanitized_u = map sanitize_f32 u
  let sanitized_v = map sanitize_f32 v
  let u_norm0 = f32.sqrt (f32.sum (map (\value -> value * value) sanitized_u))
  let v_norm0 = f32.sqrt (f32.sum (map (\value -> value * value) sanitized_v))
  let init_u_value = if vocab_size > 0 then 1f32 / f32.sqrt (f32.i64 vocab_size) else 0f32
  let init_v_value = if dim > 0 then 1f32 / f32.sqrt (f32.i64 dim) else 0f32
  let u_start = (if u_norm0 > 1e-12f32 then sanitized_u else replicate vocab_size init_u_value) :> [vocab_size]f32
  let v_start = (if v_norm0 > 1e-12f32 then sanitized_v else replicate dim init_v_value) :> [dim]f32
  let (final_u, final_v) =
    loop (ua, va) = (u_start, v_start) for loop_k < i64.max 1 power_iters do
      let _ = loop_k
      let raw_v = map (\column -> f32.sum (map2 (*) column ua)) weight_t
      let raw_v_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_v))
      let seeded_v = (if raw_v_norm > 1e-12f32 then map (/ raw_v_norm) raw_v else va) :> [dim]f32
      let raw_u = map (\row -> f32.sum (map2 (*) row seeded_v)) safe_weight
      let u_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_u))
      let next_u = (map (/ f32.max u_norm 1e-12f32) raw_u) :> [vocab_size]f32
      let refined_v = map (\column -> f32.sum (map2 (*) column next_u)) weight_t
      let refined_norm = f32.sqrt (f32.sum (map (\value -> value * value) refined_v))
      let next_v = (map (/ f32.max refined_norm 1e-12f32) refined_v) :> [dim]f32
      in (next_u, next_v)
  let final_u_sized = final_u :> [vocab_size]f32
  let final_v_sized = final_v :> [dim]f32
  let projected = map (\row -> f32.sum (map2 (*) row final_v_sized)) safe_weight
  let sigma = sanitize_f32 (f32.abs (f32.sum (map2 (*) final_u_sized projected)))
  let safe_target = f32.max target 1e-6f32
  let scale = if sigma > safe_target then safe_target / sigma else 1f32
  let normalized = map (map (* scale)) safe_weight
  in (normalized, copy final_u_sized, copy final_v_sized, sigma, sigma * scale)

let graph_derive_qubit_states [n] (hashes: [n]u64) : ([n]f32, [n]f32, [n]f32, [n]f32) =
  let pi = 3.14159265358979323846f32
  let two_pi = 2f32 * pi
  let inv_m = 1f32 / 1000000f32
  let secondary = map splitmix64 hashes
  let raw_re_a = map (\h -> f32.cos (two_pi * f32.u64 (h % 1000000u64) * inv_m)) hashes
  let raw_im_a = map (\h -> f32.sin (two_pi * f32.u64 ((h >> 20u64) % 1000000u64) * inv_m)) hashes
  let raw_re_b = map (\h -> f32.cos (two_pi * f32.u64 ((h >> 40u64) % 1000000u64) * inv_m)) hashes
  let raw_im_b = map (\s -> f32.sin (two_pi * f32.u64 (s % 1000000u64) * inv_m)) secondary
  let norms = map4 (\ra ia rb ib ->
    let s = ra * ra + ia * ia + rb * rb + ib * ib
    in if s > 1e-30f32 then f32.sqrt s else 1f32) raw_re_a raw_im_a raw_re_b raw_im_b
  in (map2 (/) raw_re_a norms, map2 (/) raw_im_a norms, map2 (/) raw_re_b norms, map2 (/) raw_im_b norms)

entry graph_batch_encode [n] (data_hashes: [n]u64) (seed: u64) : ([]u64, []f32, []f32, []f32, []f32, []i64, []i64) =
  let seed_mix = splitmix64 seed
  let mixed = map (\h -> splitmix64 (h ^ seed_mix)) data_hashes
  let (re_a, im_a, re_b, im_b) = graph_derive_qubit_states mixed
  let ne = n * 3
  let edge_srcs = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i else -1i64)
  let edge_tgts = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i - pred_k - 1 else -1i64)
  in (copy data_hashes, re_a, im_a, re_b, im_b, edge_srcs, edge_tgts)

entry embedding_sum_squares [vocab_size][dim] (source: [vocab_size][dim]f16) : f32 =
  let squared = map (\row ->
    map (\v ->
      let x = sanitize_f32 (f32.f16 v)
      in x * x) row) source
  let total = f32.sum (flatten squared)
  in sanitize_f32 total

entry rsf_stack_forward [batch_size][seq_len][half][num_layers]
  (inputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (clip_min: f16) (clip_max: f16)
  : *[batch_size][seq_len][half*2]f16 =
  let raw_min = f32.f16 clip_min
  let raw_max = f32.f16 clip_max
  let clip_min_f32 = f32.min raw_min raw_max
  let clip_max_f32 = f32.max raw_min raw_max
  let flat = flatten inputs
  let out_rows = map (\row ->
    let row_f32 = map f32.f16 row
    let result = loop cur = row_f32 for l < num_layers do
      rsf_stack_coupling_row cur weights_s[l] weights_t[l] clip_min_f32 clip_max_f32
    in map (\v -> f16.f32 (clamp_f16_value v)) result) flat
  in copy (unflatten out_rows :> [batch_size][seq_len][half*2]f16)

entry rsf_stack_inverse [batch_size][seq_len][half][num_layers]
  (outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (clip_min: f16) (clip_max: f16)
  : *[batch_size][seq_len][half*2]f16 =
  let raw_min = f32.f16 clip_min
  let raw_max = f32.f16 clip_max
  let clip_min_f32 = f32.min raw_min raw_max
  let clip_max_f32 = f32.max raw_min raw_max
  let flat = flatten outputs
  let out_rows = map (\row ->
    let row_f32 = map f32.f16 row
    let result = loop cur = row_f32 for i < num_layers do
      let l = num_layers - 1 - i
      in rsf_stack_invert_row cur weights_s[l] weights_t[l] clip_min_f32 clip_max_f32
    in map (\v -> f16.f32 (clamp_f16_value v)) result) flat
  in copy (unflatten out_rows :> [batch_size][seq_len][half*2]f16)

entry rsf_stack_backward_gradients_fused [batch_size][seq_len][half][num_layers]
  (final_outputs: [batch_size][seq_len][half*2]f16)
  (targets: [batch_size][seq_len][half*2]f16)
  (originals: [batch_size][seq_len][half*2]f16)
  (lengths: [batch_size]i64)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (grad_mean: bool)
  (gradient_scale: f32)
  (clip_min: f32)
  (clip_max: f32)
  (reconstruction_alpha: f32)
  (forward_scale: f32)
  (logdet_weight: f32)
  : (*[num_layers][half][2]f32, *[num_layers][half][2]f32,
     *[batch_size][seq_len][half*2]f16,
     f32, f32, f32) =
  let d2 = half * 2
  let safe_clip_min = f32.min clip_min clip_max
  let safe_clip_max = f32.max clip_min clip_max
  let flat_final = flatten final_outputs
  let flat_targets = flatten targets
  let flat_orig = flatten originals
  let limits = map (\length -> i64.max 0 (i64.min seq_len length)) lengths
  let valid_tokens = i64.sum limits
  let count_elements = if valid_tokens > 0 then valid_tokens * d2 else 1
  let count_elements_f32 = f32.i64 count_elements
  let count_tokens_f32 = f32.max 1f32 (f32.i64 valid_tokens)
  let gradient_element_divisor = if grad_mean then count_elements_f32 else 1f32
  let gradient_token_divisor = if grad_mean then count_tokens_f32 else 1f32
  let ld_shift = sanitize_f32 (logdet_weight / gradient_token_divisor)
  let active_indices = filter (\t ->
    let b = t / seq_len
    let j = t % seq_len
    in j < limits[b]) (iota (batch_size * seq_len))
  let active_final = map (\t -> flat_final[t]) active_indices
  let active_targets = map (\t -> flat_targets[t]) active_indices
  let active_orig = map (\t -> flat_orig[t]) active_indices
  let initial_grads = map2 (\y t ->
    map2 (\yv tv ->
      let diff = f32.f16 yv - f32.f16 tv
      let safe_diff = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in 2f32 * safe_diff / gradient_element_divisor) y t) active_final active_targets
  let y_start = map (map (\v -> sanitize_f32 (f32.f16 v))) active_final
  let gs_zero = replicate half (replicate 2 0f32)
  let (gs_stack, gt_stack, x_stack, g_stack, ld_stack) =
    loop (gs_acc, gt_acc, y_all, g_all, ld_all) =
      (replicate num_layers (copy gs_zero),
       replicate num_layers (copy gs_zero),
       y_start,
       initial_grads,
       map (\_ -> 0f32) y_start)
    for i < num_layers do
      let l = num_layers - 1 - i
      let ws = weights_s[l]
      let wt = weights_t[l]
      let per_tok = map2 (\y_row g_row ->
        let g_masked = map2 (\gv yv ->
          if f32.abs yv >= 60000f32 then 0f32 else sanitize_f32 gv) g_row y_row
        let y1p = y_row[0:half] :> [half]f32
        let y2p = y_row[half:d2] :> [half]f32
        let g1p = g_masked[0:half] :> [half]f32
        let g2p = g_masked[half:d2] :> [half]f32
        let u1 = map2 (\a b -> (a + b) * oftb_scale_f32) y1p y2p
        let u2 = map2 (\a b -> (b - a) * oftb_scale_f32) y1p y2p
        let h1 = map2 (\a b -> (a + b) * oftb_scale_f32) g1p g2p
        let h2 = map2 (\a b -> (b - a) * oftb_scale_f32) g1p g2p
        let dy1_total = map (\j ->
          h1[j] + h2[j] * f32.f16 wt[j][0]) (iota half)
        let x2 = map (\dd ->
          let trans = f32.f16 wt[dd][0] * u1[dd] + f32.f16 wt[dd][1]
          let safe_trans = sanitize_f32 trans
          in u2[dd] - safe_trans) (iota half)
        let pre_scale = map (\dd ->
          f32.f16 ws[dd][0] * x2[dd] + f32.f16 ws[dd][1]) (iota half)
        let clipped = map (\p -> f32.max safe_clip_min (f32.min safe_clip_max p)) pre_scale
        let scale = map f32.exp clipped
        let x1 = map2 (\u_j s_j -> sanitize_f32 (u_j / s_j)) u1 scale
        let dx1 = map2 (\dt_j s_j -> sanitize_f32 (dt_j * s_j)) dy1_total scale
        let ds = map3 (\p dt_j u_j ->
          if p >= safe_clip_min && p <= safe_clip_max
          then sanitize_f32 (dt_j * u_j - ld_shift)
          else 0f32) pre_scale dy1_total u1
        let dx2 = map (\j ->
          sanitize_f32 (h2[j] + ds[j] * f32.f16 ws[j][0])) (iota half)
        let y_next = (x1 ++ x2) :> [half*2]f32
        let g_next = (dx1 ++ dx2) :> [half*2]f32
        let ld_tok = f32.sum clipped
        in (ds, h2, x2, u1, y_next, g_next, ld_tok)) y_all g_all
      let ds_columns = transpose (map (\(ds, _, _, _, _, _, _) -> ds) per_tok)
      let h2_columns = transpose (map (\(_, h2, _, _, _, _, _) -> h2) per_tok)
      let x2_columns = transpose (map (\(_, _, x2, _, _, _, _) -> x2) per_tok)
      let u1_columns = transpose (map (\(_, _, _, u1, _, _, _) -> u1) per_tok)
      let gs_l_total = map2 (\ds_column x2_column ->
        [f32.sum (map2 (*) ds_column x2_column), f32.sum ds_column] :> [2]f32) ds_columns x2_columns
      let gt_l_total = map2 (\h2_column u1_column ->
        [f32.sum (map2 (*) h2_column u1_column), f32.sum h2_column] :> [2]f32) h2_columns u1_columns
      let y_next_raw = map (\(_, _, _, _, y_next, _, _) -> y_next) per_tok
      let y_next_all = if i < num_layers - 1
                       then map (map clamp_f16_value) y_next_raw
                       else y_next_raw
      let g_next_all = map (\(_, _, _, _, _, g_next, _) -> g_next) per_tok
      let ld_next = map2 (+) ld_all (map (\(_, _, _, _, _, _, ld_tok) -> ld_tok) per_tok)
      in (gs_acc with [l] = gs_l_total,
          gt_acc with [l] = gt_l_total,
          y_next_all,
          g_next_all,
          ld_next)
  let gs_normalized = map (map (map (\value -> sanitize_f32 (value * gradient_scale)))) gs_stack
  let gt_normalized = map (map (map (\value -> sanitize_f32 (value * gradient_scale)))) gt_stack
  let loss_total = reduce (+) 0f32 (map2 (\y t ->
    f32.sum (map2 (\yv tv ->
      let diff = f32.f16 yv - f32.f16 tv
      let safe = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in safe * safe) y t)) active_final active_targets)
  let loss = loss_total / count_elements_f32
  let recon_total = reduce (+) 0f32 (map2 (\x_row o_row ->
    f32.sum (map2 (\xv ov ->
      let diff = xv - f32.f16 ov
      let safe = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in safe * safe) x_row o_row)) x_stack active_orig)
  let recon_loss = recon_total / count_elements_f32
  let logdet_total = reduce (+) 0f32 ld_stack
  let logdet_mean = sanitize_f32 (logdet_total / count_tokens_f32)
  let active_input_delta = map3 (\g_row x_row o_row ->
    map3 (\gv xv ov ->
      let base = forward_scale * gradient_scale * sanitize_f32 gv
      let diff = xv - f32.f16 ov
      let safe_diff = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      let combined = sanitize_f32 (base + reconstruction_alpha * 2f32 * safe_diff / gradient_element_divisor)
      in f16.f32 (f32.max (-65504f32) (f32.min 65504f32 combined))) g_row x_row o_row
    ) g_stack x_stack active_orig
  let zero_delta = replicate (batch_size * seq_len) (replicate (half * 2) 0f16)
  let input_delta = scatter zero_delta active_indices active_input_delta
  let input_delta_3d = unflatten input_delta :> [batch_size][seq_len][half*2]f16
  in (copy gs_normalized, copy gt_normalized, input_delta_3d,
      f32.max 0f32 loss, f32.max 0f32 recon_loss, logdet_mean)