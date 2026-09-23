const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const tensor = @import("../core/tensor.zig");
const Tensor = tensor.Tensor;
const memory = @import("../core/memory.zig");
const core_io = @import("../core/io.zig");
const accel = @import("../hw/accel/accel_interface.zig");
const OFTB = @import("oftb.zig").OFTB;
const types = @import("../core/types.zig");
const Thread = std.Thread;
const LAYER_TARGET_SPECTRAL_NORM: f32 = 0.9;
const WEIGHT_COLUMN: usize = 0;
const BIAS_COLUMN: usize = 1;
const GPU_VALIDATION_SEED: u64 = 0x5D9F_1C33_A17B_0E45;
const GPU_VALIDATION_BATCH: usize = 2;
const MODEL_CROSS_CHECK_ABS_TOL: f32 = 5.0e-2;
const MODEL_CROSS_CHECK_REL_TOL: f32 = 5.0e-2;
const SAVE_VERSION: u32 = 7;
const SAVE_VERSION_LEGACY: u32 = 6;
const TENSOR_RANK_TAG: u64 = 2;
const coupling_width: usize = accel.rsf_coupling_width;
comptime {
    if (coupling_width != 2) {
        @compileError("rsf.zig implements an affine coupling with exactly one weight column and one bias column; accel.rsf_coupling_width must be 2");
    }
    if (BIAS_COLUMN >= coupling_width or WEIGHT_COLUMN >= coupling_width) {
        @compileError("rsf.zig coupling column indices must be inside the coupling width");
    }
}
fn scratchAllocator() Allocator {
    return std.heap.smp_allocator;
}
fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.Overflow;
}
fn checkedMulU64(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch return error.Overflow;
}
fn checkedAddU64(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return error.Overflow;
}
fn checkedCastU64ToUsize(v: u64) !usize {
    if (v > std.math.maxInt(usize)) return error.TooLarge;
    return @intCast(v);
}
fn constrainSpectralNorm(weight: *Tensor, rows: usize, cols: usize, target: f32) !void {
    if (!std.math.isFinite(target) or !(target > 0.0)) return error.InvalidConfig;
    if (rows == 0 or cols == 0) return error.InvalidDimension;
    if (cols != coupling_width) return error.ShapeMismatch;
    const expected = try checkedMul(rows, cols);
    if (weight.data.len < expected) return error.DataLengthMismatch;
    try ensureFiniteSlice(weight.data[0..expected]);
    const sigma = try tensor.constrainCouplingSpectralNorm(weight.data[0..expected], rows, target);
    const sigma_f32: f32 = @floatCast(sigma);
    if (!std.math.isFinite(sigma_f32)) return error.NonFinite;
}

pub fn exactSpectralNormRank2(weight: []const f32, dim: usize) !f32 {
    if (dim == 0) return error.InvalidDimension;
    const expected = try checkedMul(dim, coupling_width);
    if (weight.len < expected) return error.DataLengthMismatch;
    const sigma = try tensor.exactSpectralNormRank2(weight[0..expected], dim);
    const sigma_f32: f32 = @floatCast(sigma);
    if (!std.math.isFinite(sigma_f32)) return error.NonFinite;
    return sigma_f32;
}

pub fn layerTargetSpectralNorm() f32 {
    return LAYER_TARGET_SPECTRAL_NORM;
}

pub fn diffusionLayoutForRowLen(row_len: usize) !types.RSFDiffusionLayout {
    const layout = types.rsfDiffusionLayout(row_len) orelse return error.InvalidConfig;
    if (!tensor.diffusionLayoutIsApplicable(row_len, layout)) return error.InvalidConfig;
    return layout;
}

pub const RSFLayerConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    seed_offset: u64 = 0,
    grad_mean: bool = true,
};
pub const RSFConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    grad_mean: bool = true,
    max_dim: usize = 1 << 20,
    max_layers: usize = 1 << 20,
    global_diffusion: bool = true,
};
fn validateClipRange(clip_min: f32, clip_max: f32) !void {
    if (!std.math.isFinite(clip_min) or !std.math.isFinite(clip_max)) return error.NonFinite;
    if (!(clip_min < clip_max)) return error.InvalidConfig;
    if (clip_max > 20.0 or clip_min < -20.0) return error.InvalidConfig;
}
fn validateComparisonTolerances(abs_tol: f32, rel_tol: f32) !void {
    if (!std.math.isFinite(abs_tol) or !std.math.isFinite(rel_tol)) return error.InvalidTolerance;
    if (abs_tol < 0.0 or rel_tol < 0.0) return error.InvalidTolerance;
}
fn validateTensor2D(t: *const Tensor) !void {
    if (t.shape.dims.len != 2) return error.ShapeMismatch;
    const expected = try checkedMul(t.shape.dims[0], t.shape.dims[1]);
    if (t.data.len != expected) return error.DataLengthMismatch;
}
fn validateTensor2DShape(t: *const Tensor, rows: usize, cols: usize) !void {
    if (t.shape.dims.len != 2 or t.shape.dims[0] != rows or t.shape.dims[1] != cols) return error.ShapeMismatch;
    const expected = try checkedMul(rows, cols);
    if (t.data.len != expected) return error.DataLengthMismatch;
}
fn tensorHasShape(t: *const Tensor, rows: usize, cols: usize) bool {
    return t.shape.dims.len == 2 and t.shape.dims[0] == rows and t.shape.dims[1] == cols;
}
fn tensorsSameShape(a: *const Tensor, b: *const Tensor) bool {
    return a.shape.dims.len == 2 and b.shape.dims.len == 2 and a.shape.dims[0] == b.shape.dims[0] and a.shape.dims[1] == b.shape.dims[1];
}
fn ensureFiniteSlice(data: []const f32) !void {
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
    }
}
fn zeroTensor(t: *Tensor) void {
    @memset(t.data, @as(f32, 0.0));
}
fn tensorsOverlap(a: *const Tensor, b: *const Tensor) bool {
    if (a.data.len == 0 or b.data.len == 0) return false;
    const a_start: usize = @intFromPtr(a.data.ptr);
    const b_start: usize = @intFromPtr(b.data.ptr);
    const a_bytes = std.math.mul(usize, a.data.len, @sizeOf(f32)) catch return true;
    const b_bytes = std.math.mul(usize, b.data.len, @sizeOf(f32)) catch return true;
    const a_end = std.math.add(usize, a_start, a_bytes) catch return true;
    const b_end = std.math.add(usize, b_start, b_bytes) catch return true;
    return a_start < b_end and b_start < a_end;
}
fn tensorClone(allocator: Allocator, src: *const Tensor) !Tensor {
    try validateTensor2D(src);
    var dst = try Tensor.init(allocator, &.{ src.shape.dims[0], src.shape.dims[1] });
    errdefer dst.deinit();
    @memcpy(dst.data, src.data);
    return dst;
}
fn valuesWithinTolerance(a: f32, b: f32, abs_tol: f32, rel_tol: f32) bool {
    if (!std.math.isFinite(a) or !std.math.isFinite(b)) return false;
    const diff = @abs(a - b);
    const scale = @max(@abs(a), @abs(b));
    return diff <= abs_tol + rel_tol * scale;
}
fn tensorAllCloseEq(a: *const Tensor, b: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
    try validateComparisonTolerances(abs_tol, rel_tol);
    try validateTensor2D(a);
    try validateTensor2D(b);
    if (!tensorsSameShape(a, b)) return false;
    var i: usize = 0;
    while (i < a.data.len) : (i += 1) {
        if (!valuesWithinTolerance(a.data[i], b.data[i], abs_tol, rel_tol)) return false;
    }
    return true;
}
fn validateModelConfigValues(dim: usize, num_layers: usize, cfg: RSFConfig) !void {
    if (dim == 0) return error.InvalidDimension;
    if (num_layers == 0) return error.InvalidLayerCount;
    try validateClipRange(cfg.clip_min, cfg.clip_max);
    if (cfg.max_dim == 0 or cfg.max_layers == 0) return error.InvalidConfig;
    if (dim > cfg.max_dim or num_layers > cfg.max_layers) return error.InvalidConfig;
    if (cfg.global_diffusion) _ = try diffusionLayoutForRowLen(try checkedMul(dim, 2));
}

fn initOFTBForConfig(dim: usize, global_diffusion: bool) !OFTB {
    if (global_diffusion) {
        _ = try diffusionLayoutForRowLen(try checkedMul(dim, 2));
        return OFTB.initWithDiffusion(dim, true);
    }
    return OFTB.init(dim);
}
const LayerCore = struct {
    s_weight: Tensor,
    t_weight: Tensor,
    s_weight_grad: ?Tensor,
    t_weight_grad: ?Tensor,
    dim: usize,
    allocator: Allocator,
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,
    gn_block: ?[]f32,
    model_id: u64,
    layer_index: usize,
    rwlock: Thread.RwLock,
    fn initOwned(allocator: Allocator, dim: usize, config: RSFLayerConfig) !LayerCore {
        if (dim == 0) return error.InvalidDimension;
        try validateClipRange(config.clip_min, config.clip_max);
        _ = try checkedMul(dim, coupling_width);
        _ = try checkedMul(dim, 2);
        const fan_in: f32 = @floatFromInt(dim);
        const fan_out: f32 = @floatFromInt(dim);
        const fan_sum = fan_in + fan_out;
        if (!std.math.isFinite(fan_sum) or !(fan_sum > 0.0)) return error.InvalidDimension;
        const xavier_bound: f32 = @sqrt(6.0 / fan_sum);
        if (!std.math.isFinite(xavier_bound) or !(xavier_bound > 0.0)) return error.InvalidDimension;
        const weight_shape = [_]usize{ dim, coupling_width };
        const seed1 = try checkedAddU64(42, config.seed_offset);
        const seed2 = try checkedAddU64(43, config.seed_offset);
        var s_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed1);
        errdefer s_w.deinit();
        var t_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed2);
        errdefer t_w.deinit();
        for (0..dim) |d| {
            s_w.data[d * coupling_width + BIAS_COLUMN] = 0.0;
            t_w.data[d * coupling_width + BIAS_COLUMN] = 0.0;
        }
        try constrainSpectralNorm(&s_w, dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        try constrainSpectralNorm(&t_w, dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        return LayerCore{
            .s_weight = s_w,
            .t_weight = t_w,
            .s_weight_grad = null,
            .t_weight_grad = null,
            .dim = dim,
            .allocator = allocator,
            .clip_min = config.clip_min,
            .clip_max = config.clip_max,
            .grad_mean = config.grad_mean,
            .gn_block = null,
            .model_id = 0,
            .layer_index = 0,
            .rwlock = .{},
        };
    }
    fn deinitOwned(self: *LayerCore) void {
        self.s_weight.deinit();
        self.t_weight.deinit();
        if (self.s_weight_grad) |*g| g.deinit();
        if (self.t_weight_grad) |*g| g.deinit();
        if (self.gn_block) |blk| self.allocator.free(blk);
        self.s_weight_grad = null;
        self.t_weight_grad = null;
        self.gn_block = null;
    }
    pub fn ensureGradients(self: *LayerCore) !void {
        const need_swg = self.s_weight_grad == null;
        const need_twg = self.t_weight_grad == null;
        if (!(need_swg or need_twg)) return;
        const weight_shape = [_]usize{ self.dim, coupling_width };
        var swg_new: ?Tensor = null;
        var twg_new: ?Tensor = null;
        errdefer {
            if (swg_new) |*t| t.deinit();
            if (twg_new) |*t| t.deinit();
        }
        if (need_swg) swg_new = try Tensor.zeros(self.allocator, &weight_shape);
        if (need_twg) twg_new = try Tensor.zeros(self.allocator, &weight_shape);
        if (swg_new) |t| self.s_weight_grad = t;
        if (twg_new) |t| self.t_weight_grad = t;
        swg_new = null;
        twg_new = null;
    }
    fn zeroGradients(self: *LayerCore) void {
        if (self.s_weight_grad) |*g| zeroTensor(g);
        if (self.t_weight_grad) |*g| zeroTensor(g);
    }
    fn gradientSquaredSum(self: *const LayerCore) !f64 {
        var total: f64 = 0.0;
        if (self.s_weight_grad) |g| {
            try ensureFiniteSlice(g.data);
            for (g.data) |v| total += @as(f64, v) * @as(f64, v);
        }
        if (self.t_weight_grad) |g| {
            try ensureFiniteSlice(g.data);
            for (g.data) |v| total += @as(f64, v) * @as(f64, v);
        }
        return total;
    }
    fn applyGradientStep(self: *LayerCore, learning_rate: f32) !void {
        if (!std.math.isFinite(learning_rate)) return error.NonFinite;
        if (!(learning_rate > 0.0)) return error.InvalidLearningRate;
        const s_grad: *Tensor = if (self.s_weight_grad) |*g| g else return error.NoGradients;
        const t_grad: *Tensor = if (self.t_weight_grad) |*g| g else return error.NoGradients;
        try validateTensor2DShape(&self.s_weight, self.dim, coupling_width);
        try validateTensor2DShape(&self.t_weight, self.dim, coupling_width);
        try validateTensor2DShape(s_grad, self.dim, coupling_width);
        try validateTensor2DShape(t_grad, self.dim, coupling_width);
        try ensureFiniteSlice(s_grad.data);
        try ensureFiniteSlice(t_grad.data);
        var s_next = try tensorClone(self.allocator, &self.s_weight);
        errdefer s_next.deinit();
        var t_next = try tensorClone(self.allocator, &self.t_weight);
        errdefer t_next.deinit();
        var i: usize = 0;
        while (i < s_next.data.len) : (i += 1) {
            const updated = s_next.data[i] - learning_rate * s_grad.data[i];
            if (!std.math.isFinite(updated)) return error.NonFinite;
            s_next.data[i] = updated;
        }
        i = 0;
        while (i < t_next.data.len) : (i += 1) {
            const updated = t_next.data[i] - learning_rate * t_grad.data[i];
            if (!std.math.isFinite(updated)) return error.NonFinite;
            t_next.data[i] = updated;
        }
        try constrainSpectralNorm(&s_next, self.dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        try constrainSpectralNorm(&t_next, self.dim, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
        try ensureFiniteSlice(s_next.data);
        try ensureFiniteSlice(t_next.data);
        self.s_weight.deinit();
        self.s_weight = s_next;
        self.t_weight.deinit();
        self.t_weight = t_next;
    }
    fn validatePair(self: *const LayerCore, a: *const Tensor, b: *const Tensor) !usize {
        try validateTensor2D(a);
        try validateTensor2D(b);
        if (a.shape.dims[1] != self.dim or b.shape.dims[1] != self.dim) return error.ShapeMismatch;
        if (a.shape.dims[0] != b.shape.dims[0]) return error.ShapeMismatch;
        const batch_size = a.shape.dims[0];
        if (batch_size == 0) return error.InvalidBatchSize;
        _ = try checkedMul(batch_size, self.dim);
        return batch_size;
    }
    fn couplingParams(self: *const LayerCore) !tensor.RSFCouplingParams {
        const expected = try checkedMul(self.dim, coupling_width);
        if (self.s_weight.data.len < expected) return error.DataLengthMismatch;
        if (self.t_weight.data.len < expected) return error.DataLengthMismatch;
        return tensor.RSFCouplingParams.init(self.s_weight.data[0..expected], self.t_weight.data[0..expected], self.dim, self.clip_min, self.clip_max);
    }
    fn couplingForwardRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !void {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        _ = try tensor.couplingForwardHalves(params, row[0..dim], row[dim..total], scale, trans);
    }
    fn couplingForwardLogDetRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !f32 {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        const logdet = try tensor.couplingForwardHalves(params, row[0..dim], row[dim..total], scale, trans);
        const logdet_f32: f32 = @floatCast(logdet);
        if (!std.math.isFinite(logdet_f32)) return error.NonFinite;
        return logdet_f32;
    }
    fn couplingInverseRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !void {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        _ = try tensor.couplingInverseHalves(params, row[0..dim], row[dim..total], scale, trans);
    }
    fn couplingInverseLogDetRow(self: *const LayerCore, row: []f32, scale: []f32, trans: []f32) !f32 {
        const dim = self.dim;
        const total = try checkedMul(dim, 2);
        if (row.len != total) return error.DataLengthMismatch;
        if (scale.len < dim or trans.len < dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        const logdet = try tensor.couplingInverseHalves(params, row[0..dim], row[dim..total], scale, trans);
        const logdet_f32: f32 = @floatCast(logdet);
        if (!std.math.isFinite(logdet_f32)) return error.NonFinite;
        return logdet_f32;
    }
    fn forwardInPlace(self: *const LayerCore, x1: *Tensor, x2: *Tensor, scale: []f32, trans: []f32) !void {
        if (scale.len < self.dim or trans.len < self.dim) return error.DataLengthMismatch;
        if (tensorsOverlap(x1, x2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(x1, x2);
        const params = try self.couplingParams();
        _ = try tensor.couplingForwardStrided(params, x1.data, x2.data, batch_size, self.dim, self.dim, scale, trans);
    }
    fn inverseInPlace(self: *const LayerCore, y1: *Tensor, y2: *Tensor, scale: []f32, trans: []f32) !void {
        if (scale.len < self.dim or trans.len < self.dim) return error.DataLengthMismatch;
        if (tensorsOverlap(y1, y2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(y1, y2);
        const params = try self.couplingParams();
        _ = try tensor.couplingInverseStrided(params, y1.data, y2.data, batch_size, self.dim, self.dim, scale, trans);
    }
    fn ensureGaussNewton(self: *LayerCore) !void {
        if (self.gn_block != null) return;
        const len = try checkedMul(self.dim, 3);
        const blk = try self.allocator.alloc(f32, len);
        @memset(blk, 0.0);
        self.gn_block = blk;
    }
    fn zeroGaussNewton(self: *LayerCore) void {
        if (self.gn_block) |blk| @memset(blk, 0.0);
    }
    fn recordGaussNewtonRow(self: *LayerCore, x2_row: []const f32, params: tensor.RSFCouplingParams, logdet_adjoint: f32) !void {
        if (logdet_adjoint == 0.0) return;
        try self.ensureGaussNewton();
        const blk = self.gn_block orelse return error.NoGaussNewtonBlock;
        const dim = self.dim;
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            const raw = params.scaleWeight(d) * x2_row[d] + params.scaleBias(d);
            const active: f32 = if (tensor.couplingSaturates(raw, params.clip_min, params.clip_max)) 0.0 else logdet_adjoint;
            const g_w = active * x2_row[d];
            blk[d * 3 + 0] += g_w * g_w;
            blk[d * 3 + 1] += g_w * active;
            blk[d * 3 + 2] += active * active;
        }
    }
    fn backwardFromInputsRow(
        self: *LayerCore,
        x1_row: []const f32,
        x2_row: []const f32,
        dy1_row: []const f32,
        dy2_row: []const f32,
        dx1_row_out: []f32,
        dx2_row_out: []f32,
        y1_buf: []f32,
        scale_buf: []f32,
        ds_weight_buf: []f32,
        dt_weight_buf: []f32,
        grad_scale: f32,
        logdet_adjoint: f32,
    ) !void {
        const dim = self.dim;
        if (!std.math.isFinite(grad_scale)) return error.NonFinite;
        if (!std.math.isFinite(logdet_adjoint)) return error.NonFinite;
        if (x1_row.len != dim or x2_row.len != dim) return error.ShapeMismatch;
        if (dy1_row.len != dim or dy2_row.len != dim) return error.ShapeMismatch;
        if (dx1_row_out.len != dim or dx2_row_out.len != dim) return error.ShapeMismatch;
        if (y1_buf.len != dim or scale_buf.len != dim) return error.DataLengthMismatch;
        const params = try self.couplingParams();
        const expected = try checkedMul(dim, coupling_width);
        if (ds_weight_buf.len < expected or dt_weight_buf.len < expected) return error.DataLengthMismatch;
        @memset(ds_weight_buf[0..expected], 0.0);
        @memset(dt_weight_buf[0..expected], 0.0);
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            const scale = tensor.couplingScaleFromInput(params, x2_row[d], d);
            scale_buf[d] = scale;
            y1_buf[d] = x1_row[d] * scale;
        }
        _ = try tensor.couplingBackwardHalves(
            params,
            x1_row,
            x2_row,
            y1_buf,
            dy1_row,
            dy2_row,
            logdet_adjoint,
            ds_weight_buf[0..expected],
            dt_weight_buf[0..expected],
            dx1_row_out,
            dx2_row_out,
        );
        try self.recordGaussNewtonRow(x2_row, params, logdet_adjoint);
        if (self.s_weight_grad) |*swg| {
            if (swg.data.len >= expected) {
                d = 0;
                while (d < expected) : (d += 1) swg.data[d] += grad_scale * ds_weight_buf[d];
            }
        }
        if (self.t_weight_grad) |*twg| {
            if (twg.data.len >= expected) {
                d = 0;
                while (d < expected) : (d += 1) twg.data[d] += grad_scale * dt_weight_buf[d];
            }
        }
    }
};
const LayerBindings = struct {
    s_weight: types.RSFBinding,
    t_weight: types.RSFBinding,
    gradient: types.RSFBinding,
    fisher_block: types.RSFBinding,
};

fn layerBindings(model_id: u64, layer_index: usize, dim: usize) LayerBindings {
    return .{
        .s_weight = types.RSFBinding.layer(.layer_weight_s, model_id, layer_index, dim),
        .t_weight = types.RSFBinding.layer(.layer_weight_t, model_id, layer_index, dim),
        .gradient = types.RSFBinding.layer(.gradient, model_id, layer_index, dim),
        .fisher_block = types.RSFBinding.layer(.fisher_block, model_id, layer_index, dim),
    };
}

fn checkedLayerIndex(core: *const RSFCore, layer: usize) !*LayerCore {
    const count = try checkedModelLayerCount(core);
    if (layer >= count) return error.LayerIndexOutOfBounds;
    return &core.layers[layer];
}

fn assignLayerBindings(core: *RSFCore, model_id: u64) void {
    for (core.layers, 0..) |*layer, index| {
        layer.model_id = model_id;
        layer.layer_index = index;
    }
}

const LayerRegistryEntry = struct {
    core: *LayerCore,
    active_ops: usize,
    destroyed: bool,
};
fn maybeShrinkRegistry(comptime EntryType: type, registry: *std.AutoHashMap(u64, EntryType)) void {
    if (registry.count() == 0) {
        registry.deinit();
        registry.* = std.AutoHashMap(u64, EntryType).init(std.heap.page_allocator);
    }
}
fn registerRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    next_id: *std.atomic.Value(u64),
    core: *CoreType,
) !u64 {
    mutex.lock();
    defer mutex.unlock();
    var id: u64 = 0;
    while (id == 0 or registry.contains(id)) {
        id = next_id.fetchAdd(1, .monotonic);
    }
    try registry.put(id, .{ .core = core, .active_ops = 0, .destroyed = false });
    return id;
}
fn acquireRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
) !*CoreType {
    if (id == 0) return error.NotInitialized;
    mutex.lock();
    defer mutex.unlock();
    const entry = registry.getPtr(id) orelse return error.NotInitialized;
    if (entry.destroyed) return error.NotInitialized;
    if (entry.active_ops == std.math.maxInt(usize)) return error.TooManyActiveOperations;
    entry.active_ops += 1;
    return entry.core;
}
fn releaseRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
    destroy_fn: *const fn (*CoreType) void,
) void {
    if (id == 0) return;
    var core_to_destroy: ?*CoreType = null;
    mutex.lock();
    if (registry.getPtr(id)) |entry| {
        if (entry.active_ops > 0) entry.active_ops -= 1;
        if (entry.destroyed and entry.active_ops == 0) {
            if (registry.fetchRemove(id)) |kv| {
                core_to_destroy = kv.value.core;
                maybeShrinkRegistry(EntryType, registry);
            }
        }
    }
    mutex.unlock();
    if (core_to_destroy) |core| destroy_fn(core);
}
fn requestDestroyRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
    destroy_fn: *const fn (*CoreType) void,
) void {
    if (id == 0) return;
    var core_to_destroy: ?*CoreType = null;
    mutex.lock();
    if (registry.getPtr(id)) |entry| {
        entry.destroyed = true;
        if (entry.active_ops == 0) {
            if (registry.fetchRemove(id)) |kv| {
                core_to_destroy = kv.value.core;
                maybeShrinkRegistry(EntryType, registry);
            }
        }
    }
    mutex.unlock();
    if (core_to_destroy) |core| destroy_fn(core);
}
fn handleId(id: u64) !u64 {
    if (id == 0) return error.NotInitialized;
    return id;
}
var g_layer_registry_mutex: Thread.Mutex = .{};
var g_layer_registry = std.AutoHashMap(u64, LayerRegistryEntry).init(std.heap.page_allocator);
var g_layer_next_id = std.atomic.Value(u64).init(1);
fn destroyLayerCore(core: *LayerCore) void {
    const allocator = core.allocator;
    core.deinitOwned();
    allocator.destroy(core);
}
fn registerLayerCore(core: *LayerCore) !u64 {
    return registerRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, &g_layer_next_id, core);
}
fn acquireLayerCore(id: u64) !*LayerCore {
    return acquireRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id);
}
fn releaseLayerCore(id: u64) void {
    releaseRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id, destroyLayerCore);
}
fn requestDestroyLayerCore(id: u64) void {
    requestDestroyRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id, destroyLayerCore);
}
pub const RSFLayer = struct {
    id: u64 = 0,
    pub fn init(allocator: Allocator, dim: usize) !RSFLayer {
        return initWithConfig(allocator, dim, .{});
    }
    pub fn initWithArena(arena: *memory.ArenaAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(arena.allocator(), dim, config);
    }
    pub fn initWithPool(pool: *memory.PoolAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(pool.allocator(), dim, config);
    }
    pub fn initWithSlab(slab: *memory.SlabAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(slab.allocator(), dim, config);
    }
    pub fn initWithBuddy(buddy: *memory.BuddyAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(buddy.allocator(), dim, config);
    }
    pub fn initWithConfig(allocator: Allocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        const core = try allocator.create(LayerCore);
        errdefer allocator.destroy(core);
        core.* = try LayerCore.initOwned(allocator, dim, config);
        errdefer core.deinitOwned();
        const id = try registerLayerCore(core);
        return RSFLayer{ .id = id };
    }
    pub fn ensureGradients(self: *RSFLayer) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try core.ensureGradients();
    }
    pub fn deinit(self: *RSFLayer) void {
        const id = self.id;
        if (id == 0) return;
        self.id = 0;
        requestDestroyLayerCore(id);
    }
    pub fn zeroGradients(self: *RSFLayer) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        core.zeroGradients();
    }
    pub fn applyGradientStep(self: *RSFLayer, learning_rate: f32) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try core.applyGradientStep(learning_rate);
    }
    pub fn gradientL2Norm(self: *const RSFLayer) !f32 {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const total = try core.gradientSquaredSum();
        const result: f32 = @floatCast(@sqrt(total));
        if (!std.math.isFinite(result)) return error.NonFinite;
        return result;
    }
    pub fn forward(self: *const RSFLayer, x1: *Tensor, x2: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.forwardInPlace(x1, x2, scale, trans);
    }
    pub fn inverse(self: *const RSFLayer, y1: *Tensor, y2: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.inverseInPlace(y1, y2, scale, trans);
    }
    pub fn verifyInvertible(self: *const RSFLayer, x1: *const Tensor, x2: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        const id = try handleId(self.id);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        var fx1 = try tensorClone(allocator, x1);
        defer fx1.deinit();
        var fx2 = try tensorClone(allocator, x2);
        defer fx2.deinit();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.forwardInPlace(&fx1, &fx2, scale, trans);
        try core.inverseInPlace(&fx1, &fx2, scale, trans);
        const ok1 = try tensorAllCloseEq(x1, &fx1, abs_tol, rel_tol);
        if (!ok1) return false;
        const ok2 = try tensorAllCloseEq(x2, &fx2, abs_tol, rel_tol);
        return ok2;
    }
};
const RSFCore = struct {
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    layers: []LayerCore,
    cfg: RSFConfig,
    rwlock: Thread.RwLock,
    gpu_accel: ?accel.RSFAccelerator,
    gpu_available: std.atomic.Value(u8),
    gpu_weight_version: u64,
    cpu_weight_version: u64,
    f16_buf: ?[]f16,
    oftb: OFTB,
};
const ModelRegistryEntry = struct {
    core: *RSFCore,
    active_ops: usize,
    destroyed: bool,
};
var g_model_registry_mutex: Thread.Mutex = .{};
var g_model_registry = std.AutoHashMap(u64, ModelRegistryEntry).init(std.heap.page_allocator);
var g_model_next_id = std.atomic.Value(u64).init(1);
fn destroyModelCore(core: *RSFCore) void {
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_available.store(0, .monotonic);
    core.oftb.deinit();
    const allocator = core.allocator;
    for (core.layers) |*layer| layer.deinitOwned();
    allocator.free(core.layers);
    allocator.destroy(core);
}
fn registerModelCore(core: *RSFCore) !u64 {
    return registerRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, &g_model_next_id, core);
}
fn acquireModelCore(id: u64) !*RSFCore {
    return acquireRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id);
}
fn releaseModelCore(id: u64) void {
    releaseRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id, destroyModelCore);
}
fn requestDestroyModelCore(id: u64) void {
    requestDestroyRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id, destroyModelCore);
}
pub fn shutdownGlobalRegistries() !void {
    g_layer_registry_mutex.lock();
    defer g_layer_registry_mutex.unlock();
    g_model_registry_mutex.lock();
    defer g_model_registry_mutex.unlock();
    if (g_layer_registry.count() != 0 or g_model_registry.count() != 0) return error.ResourcesStillActive;
    g_layer_registry.deinit();
    g_layer_registry = std.AutoHashMap(u64, LayerRegistryEntry).init(std.heap.page_allocator);
    g_model_registry.deinit();
    g_model_registry = std.AutoHashMap(u64, ModelRegistryEntry).init(std.heap.page_allocator);
}
fn checkedModelLayerCount(core: *const RSFCore) !usize {
    if (core.num_layers != core.layers.len) return error.InvalidModelState;
    if (core.layers.len == 0) return error.InvalidLayerCount;
    return core.layers.len;
}
fn validateModelMetadata(core: *const RSFCore) !void {
    const layer_count = try checkedModelLayerCount(core);
    try validateModelConfigValues(core.dim, layer_count, core.cfg);
    if (core.oftb.dim != core.dim) return error.InvalidModelState;
    var i: usize = 0;
    while (i < layer_count) : (i += 1) {
        const layer = &core.layers[i];
        if (layer.dim != core.dim) return error.InvalidModelState;
        if (layer.clip_min != core.cfg.clip_min or layer.clip_max != core.cfg.clip_max or layer.grad_mean != core.cfg.grad_mean) return error.InvalidConfig;
        try validateTensor2DShape(&layer.s_weight, core.dim, coupling_width);
        try validateTensor2DShape(&layer.t_weight, core.dim, coupling_width);
    }
}
fn forwardOnCore(core: *const RSFCore, x: *Tensor) !void {
    try validateTensor2D(x);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = x.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    const allocator = scratchAllocator();
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    var l: usize = 0;
    while (l < layer_count) : (l += 1) {
        const layer = &core.layers[l];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = x.data[b * dim2 .. b * dim2 + dim2];
            try layer.couplingForwardRow(row, scale, trans);
            core.oftb.forwardSliceInPlace(row);
        }
    }
}
fn inverseOnCore(core: *const RSFCore, y: *Tensor) !void {
    try validateTensor2D(y);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (y.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = y.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    const allocator = scratchAllocator();
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    var idx = layer_count;
    while (idx > 0) : (idx -= 1) {
        const layer = &core.layers[idx - 1];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = y.data[b * dim2 .. b * dim2 + dim2];
            core.oftb.inverseSliceInPlace(row);
            try layer.couplingInverseRow(row, scale, trans);
        }
    }
}
fn forwardLogDetOnCore(core: *const RSFCore, x: *Tensor, logdet_per_row: []f32) !void {
    try validateTensor2D(x);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = x.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    if (logdet_per_row.len < batch_size) return error.DataLengthMismatch;
    @memset(logdet_per_row[0..batch_size], 0.0);
    const allocator = scratchAllocator();
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    var l: usize = 0;
    while (l < layer_count) : (l += 1) {
        const layer = &core.layers[l];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = x.data[b * dim2 .. b * dim2 + dim2];
            const row_logdet = try layer.couplingForwardLogDetRow(row, scale, trans);
            logdet_per_row[b] += row_logdet;
            core.oftb.forwardSliceInPlace(row);
        }
    }
}
fn inverseLogDetOnCore(core: *const RSFCore, y: *Tensor, logdet_per_row: []f32) !void {
    try validateTensor2D(y);
    try validateModelMetadata(core);
    const layer_count = try checkedModelLayerCount(core);
    const dim2 = try checkedMul(core.dim, 2);
    if (y.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = y.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    if (logdet_per_row.len < batch_size) return error.DataLengthMismatch;
    @memset(logdet_per_row[0..batch_size], 0.0);
    const allocator = scratchAllocator();
    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);
    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);
    var idx = layer_count;
    while (idx > 0) : (idx -= 1) {
        const layer = &core.layers[idx - 1];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = y.data[b * dim2 .. b * dim2 + dim2];
            core.oftb.inverseSliceInPlace(row);
            const row_logdet = try layer.couplingInverseLogDetRow(row, scale, trans);
            logdet_per_row[b] += row_logdet;
        }
    }
}

fn backwardOnCore(core: *RSFCore, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor, logdet_weight: f32) !void {
    try validateModelMetadata(core);
    try validateTensor2D(grad_output);
    try validateTensor2D(input);
    try validateTensor2D(output);
    try validateTensor2D(grad_input_out);
    const layer_count = try checkedModelLayerCount(core);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);
    if (input.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (!tensorsSameShape(grad_output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(grad_input_out, input)) return error.ShapeMismatch;
    if (!std.math.isFinite(logdet_weight)) return error.NonFinite;
    if (tensorsOverlap(grad_input_out, grad_output)) return error.AliasedBuffers;
    if (tensorsOverlap(grad_input_out, input)) return error.AliasedBuffers;
    if (tensorsOverlap(grad_input_out, output)) return error.AliasedBuffers;
    try ensureFiniteSlice(input.data);
    try ensureFiniteSlice(output.data);
    try ensureFiniteSlice(grad_output.data);
    const batch_size = input.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;
    var li: usize = 0;
    while (li < layer_count) : (li += 1) try core.layers[li].ensureGradients();
    const grad_scale: f32 = blk: {
        if (!core.cfg.grad_mean) break :blk 1.0;
        const s = 1.0 / @as(f32, @floatFromInt(batch_size));
        break :blk if (std.math.isFinite(s)) s else 1.0;
    };
    const allocator = scratchAllocator();
    const state_len = try checkedMul(layer_count, dim2);
    const states = try allocator.alloc(f32, state_len);
    defer allocator.free(states);
    const cur = try allocator.alloc(f32, dim2);
    defer allocator.free(cur);
    const dy = try allocator.alloc(f32, dim2);
    defer allocator.free(dy);
    const dx = try allocator.alloc(f32, dim2);
    defer allocator.free(dx);
    const scale_buf = try allocator.alloc(f32, dim);
    defer allocator.free(scale_buf);
    const y1_buf = try allocator.alloc(f32, dim);
    defer allocator.free(y1_buf);
    const trans_buf = try allocator.alloc(f32, dim);
    defer allocator.free(trans_buf);
    const per_layer_params = try checkedMul(dim, coupling_width);
    const ds_weight_buf = try allocator.alloc(f32, per_layer_params);
    defer allocator.free(ds_weight_buf);
    const dt_weight_buf = try allocator.alloc(f32, per_layer_params);
    defer allocator.free(dt_weight_buf);
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        @memcpy(cur, input.data[b * dim2 .. b * dim2 + dim2]);
        var l: usize = 0;
        while (l < layer_count) : (l += 1) {
            @memcpy(states[l * dim2 .. l * dim2 + dim2], cur);
            try core.layers[l].couplingForwardRow(cur, scale_buf, trans_buf);
            core.oftb.forwardSliceInPlace(cur);
        }
        var check: usize = 0;
        while (check < dim2) : (check += 1) {
            if (!valuesWithinTolerance(cur[check], output.data[b * dim2 + check], MODEL_CROSS_CHECK_ABS_TOL, MODEL_CROSS_CHECK_REL_TOL)) return error.StateMismatch;
        }
        @memcpy(dy, grad_output.data[b * dim2 .. b * dim2 + dim2]);
        var idx = layer_count;
        while (idx > 0) : (idx -= 1) {
            core.oftb.backwardSliceInPlace(dy);
            const state = states[(idx - 1) * dim2 .. (idx - 1) * dim2 + dim2];
            try core.layers[idx - 1].backwardFromInputsRow(
                state[0..dim],
                state[dim..dim2],
                dy[0..dim],
                dy[dim..dim2],
                dx[0..dim],
                dx[dim..dim2],
                y1_buf,
                scale_buf,
                ds_weight_buf,
                dt_weight_buf,
                grad_scale,
                logdet_weight,
            );
            @memcpy(dy, dx);
        }
        @memcpy(grad_input_out.data[b * dim2 .. b * dim2 + dim2], dy);
    }
}
fn layerGPUCompatible(layer: *const LayerCore, cfg: *const RSFConfig, dim: usize) bool {
    if (layer.dim != dim) return false;
    if (layer.clip_min != cfg.clip_min or layer.clip_max != cfg.clip_max or layer.grad_mean != cfg.grad_mean) return false;
    return std.math.isFinite(layer.clip_min) and std.math.isFinite(layer.clip_max) and layer.clip_min < layer.clip_max;
}
fn modelGPUCompatible(core: *const RSFCore) bool {
    if (comptime !accel.gpu_enabled) return false;
    if (core.layers.len == 0) return false;
    if (core.num_layers != core.layers.len) return false;
    for (core.layers) |*layer| {
        if (!layerGPUCompatible(layer, &core.cfg, core.dim)) return false;
    }
    return true;
}
fn disableGPU(core: *RSFCore) void {
    core.gpu_available.store(0, .monotonic);
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_weight_version = 0;
}
fn validateF16Convertible(data: []const f32) !void {
    const max_f16: f32 = @floatCast(std.math.floatMax(f16));
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
        if (@abs(v) > max_f16) return error.NumericFailure;
    }
}
fn uploadLayerToAccel(core: *const RSFCore, layer: *const LayerCore, layer_index: usize, ga: *accel.RSFAccelerator, f16_buf: []f16) !void {
    const per_layer = try checkedMul(core.dim, coupling_width);
    if (f16_buf.len < per_layer) return error.DataLengthMismatch;
    try validateTensor2DShape(&layer.s_weight, core.dim, coupling_width);
    try validateTensor2DShape(&layer.t_weight, core.dim, coupling_width);
    try validateF16Convertible(layer.s_weight.data);
    try validateF16Convertible(layer.t_weight.data);
    var i: usize = 0;
    while (i < per_layer) : (i += 1) f16_buf[i] = @floatCast(layer.s_weight.data[i]);
    try ga.setLayerWeightsS(layer_index, f16_buf[0..per_layer], core.dim, coupling_width);
    i = 0;
    while (i < per_layer) : (i += 1) f16_buf[i] = @floatCast(layer.t_weight.data[i]);
    try ga.setLayerWeightsT(layer_index, f16_buf[0..per_layer], core.dim, coupling_width);
}
fn validateAcceleratorAgainstCPU(core: *const RSFCore, ga: *accel.RSFAccelerator) !void {
    const dim2 = try checkedMul(core.dim, 2);
    const allocator = scratchAllocator();
    var probe = try Tensor.init(allocator, &.{ GPU_VALIDATION_BATCH, dim2 });
    defer probe.deinit();
    var prng = types.PRNG.init(GPU_VALIDATION_SEED);
    for (probe.data) |*v| v.* = prng.float() * 0.5 - 0.25;
    var expected = try tensorClone(allocator, &probe);
    defer expected.deinit();
    try forwardOnCore(core, &expected);
    try ensureFiniteSlice(expected.data);
    var produced = ga.forwardFromTensor(&probe, allocator) catch return error.GPUValidationFailed;
    defer produced.deinit();
    if (!tensorHasShape(&produced, GPU_VALIDATION_BATCH, dim2)) return error.GPUValidationFailed;
    if (produced.data.len != expected.data.len) return error.GPUValidationFailed;
    try ensureFiniteSlice(produced.data);
    if (!(try tensorAllCloseEq(&expected, &produced, MODEL_CROSS_CHECK_ABS_TOL, MODEL_CROSS_CHECK_REL_TOL))) return error.GPUNumericMismatch;
}
fn syncAllLayersGPU(core: *RSFCore) !void {
    var success = false;
    defer if (!success) disableGPU(core);
    if (comptime !accel.gpu_enabled) return error.GPUUnsupportedConfiguration;
    try validateModelMetadata(core);
    if (!modelGPUCompatible(core)) return error.GPUUnsupportedConfiguration;
    const layer_count = try checkedModelLayerCount(core);
    const per_layer = try checkedMul(core.dim, coupling_width);
    for (core.layers) |*layer| {
        try ensureFiniteSlice(layer.s_weight.data);
        try ensureFiniteSlice(layer.t_weight.data);
        try validateF16Convertible(layer.s_weight.data);
        try validateF16Convertible(layer.t_weight.data);
    }
    const local_f16 = try core.allocator.alloc(f16, per_layer);
    var local_f16_owned = true;
    errdefer if (local_f16_owned) core.allocator.free(local_f16);
    const accel_model_dim = try checkedMul(core.dim, 2);
    var staged_accel = accel.RSFAccelerator.initMultiLayer(accel_model_dim, layer_count, core.allocator) catch return error.NoGPUAvailable;
    var staged_owned = true;
    errdefer if (staged_owned) staged_accel.deinit();
    try staged_accel.setClipRange(@floatCast(core.cfg.clip_min), @floatCast(core.cfg.clip_max));
    var index: usize = 0;
    while (index < layer_count) : (index += 1) {
        try uploadLayerToAccel(core, &core.layers[index], index, &staged_accel, local_f16);
    }
    try validateAcceleratorAgainstCPU(core, &staged_accel);
    if (core.gpu_accel) |*ga| ga.deinit();
    if (core.f16_buf) |buf| core.allocator.free(buf);
    core.gpu_accel = staged_accel;
    staged_owned = false;
    core.f16_buf = local_f16;
    local_f16_owned = false;
    core.gpu_weight_version = core.cpu_weight_version;
    core.gpu_available.store(1, .monotonic);
    success = true;
}
fn tryForwardGPU(core: *RSFCore, x: *Tensor) bool {
    if (comptime !accel.gpu_enabled) return false;
    if (!modelGPUCompatible(core)) {
        disableGPU(core);
        return false;
    }
    if (core.gpu_available.load(.monotonic) == 0) return false;
    if (core.gpu_weight_version != core.cpu_weight_version) return false;
    if (core.gpu_accel) |*ga| {
        if (ga.numLayers() != core.layers.len) {
            disableGPU(core);
            return false;
        }
        const allocator = scratchAllocator();
        if (ga.forwardFromTensor(x, allocator)) |result| {
            var gpu_result = result;
            defer gpu_result.deinit();
            if (!tensorHasShape(&gpu_result, x.shape.dims[0], x.shape.dims[1]) or gpu_result.data.len != x.data.len) {
                disableGPU(core);
                return false;
            }
            ensureFiniteSlice(gpu_result.data) catch {
                disableGPU(core);
                return false;
            };
            @memcpy(x.data, gpu_result.data);
            return true;
        } else |_| {
            disableGPU(core);
            return false;
        }
    }
    return false;
}
fn gpuPathSelected(core: *const RSFCore) bool {
    if (comptime !accel.gpu_enabled) return false;
    return core.gpu_available.load(.monotonic) != 0 or modelGPUCompatible(core);
}
fn forwardDispatchOnCore(core: *RSFCore, x: *Tensor) !void {
    try validateTensor2D(x);
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (x.shape.dims[0] == 0) return error.InvalidBatchSize;
    if (comptime accel.gpu_enabled) {
        if (modelGPUCompatible(core)) {
            if (tryForwardGPU(core, x)) return;
            syncAllLayersGPU(core) catch {};
            if (tryForwardGPU(core, x)) return;
        } else if (core.gpu_available.load(.monotonic) != 0 or core.gpu_accel != null or core.f16_buf != null or core.gpu_weight_version != 0) {
            disableGPU(core);
        }
    }
    try forwardOnCore(core, x);
}
fn bumpWeightVersion(core: *RSFCore) void {
    core.cpu_weight_version +%= 1;
    if (core.cpu_weight_version == 0) core.cpu_weight_version = 1;
}
fn refreshGPUAfterWeightChange(core: *RSFCore) void {
    if (comptime !accel.gpu_enabled) {
        disableGPU(core);
        return;
    }
    if (!modelGPUCompatible(core)) {
        disableGPU(core);
        return;
    }
    syncAllLayersGPU(core) catch disableGPU(core);
}
const SavedLayerSnapshot = struct {
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,
    s_weight: Tensor,
    t_weight: Tensor,
    gn_block: []f32,
};
const SavedModelSnapshot = struct {
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    cfg: RSFConfig,
    layers: []SavedLayerSnapshot,
    fn deinit(self: *SavedModelSnapshot) void {
        const layers = self.layers;
        self.layers = layers[0..0];
        for (layers) |*layer| {
            layer.s_weight.deinit();
            layer.t_weight.deinit();
            self.allocator.free(layer.gn_block);
        }
        if (layers.len != 0) self.allocator.free(layers);
    }
};
fn snapshotModelForSave(allocator: Allocator, core: *const RSFCore) !SavedModelSnapshot {
    try validateModelMetadata(core);
    const layer_count = core.layers.len;
    const layers = try allocator.alloc(SavedLayerSnapshot, layer_count);
    errdefer allocator.free(layers);
    const gn_len = try checkedMul(core.dim, 3);
    var initialized: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < initialized) : (i += 1) {
            layers[i].s_weight.deinit();
            layers[i].t_weight.deinit();
            allocator.free(layers[i].gn_block);
        }
    }
    var i: usize = 0;
    while (i < layer_count) : (i += 1) {
        const layer = &core.layers[i];
        try validateClipRange(layer.clip_min, layer.clip_max);
        try ensureFiniteSlice(layer.s_weight.data);
        try ensureFiniteSlice(layer.t_weight.data);
        var sw = try tensorClone(allocator, &layer.s_weight);
        errdefer sw.deinit();
        var tw = try tensorClone(allocator, &layer.t_weight);
        errdefer tw.deinit();
        const gn = try allocator.alloc(f32, gn_len);
        errdefer allocator.free(gn);
        @memset(gn, 0.0);
        if (layer.gn_block) |blk| {
            if (blk.len >= gn_len) {
                try ensureFiniteSlice(blk[0..gn_len]);
                @memcpy(gn, blk[0..gn_len]);
            }
        }
        layers[i] = .{
            .clip_min = layer.clip_min,
            .clip_max = layer.clip_max,
            .grad_mean = layer.grad_mean,
            .s_weight = sw,
            .t_weight = tw,
            .gn_block = gn,
        };
        initialized += 1;
    }
    return .{
        .allocator = allocator,
        .dim = core.dim,
        .num_layers = core.num_layers,
        .cfg = core.cfg,
        .layers = layers,
    };
}
pub const RSFBranch = enum { s, t };

const LatentShape = struct {
    batch: usize,
    dim: usize,
};

fn latentShapeOf(state: *const RSFLatentState) !LatentShape {
    if (state.data.shape.dims.len != 3) return error.ShapeMismatch;
    if (state.data.shape.dims[2] != 2) return error.ShapeMismatch;
    const batch = state.data.shape.dims[0];
    const dim = state.data.shape.dims[1];
    if (batch == 0) return error.InvalidBatchSize;
    if (dim == 0) return error.InvalidDimension;
    _ = try checkedMul(try checkedMul(batch, dim), 2);
    return .{ .batch = batch, .dim = dim };
}

pub const LatentHalfView = struct {
    data: []f32,
    batch: usize,
    dim: usize,
    offset: usize,

    pub fn index(self: *const LatentHalfView, b: usize, d: usize) !usize {
        if (b >= self.batch or d >= self.dim) return error.OutOfBounds;
        return (b * self.dim + d) * 2 + self.offset;
    }

    pub fn get(self: *const LatentHalfView, b: usize, d: usize) !f32 {
        return self.data[try self.index(b, d)];
    }

    pub fn set(self: *const LatentHalfView, b: usize, d: usize, value: f32) !void {
        if (!std.math.isFinite(value)) return error.NonFinite;
        self.data[try self.index(b, d)] = value;
    }

    pub fn copyRowTo(self: *const LatentHalfView, b: usize, out: []f32) !void {
        if (b >= self.batch) return error.OutOfBounds;
        if (out.len < self.dim) return error.DataLengthMismatch;
        var d: usize = 0;
        while (d < self.dim) : (d += 1) out[d] = self.data[(b * self.dim + d) * 2 + self.offset];
    }

    pub fn copyRowFrom(self: *const LatentHalfView, b: usize, src: []const f32) !void {
        if (b >= self.batch) return error.OutOfBounds;
        if (src.len < self.dim) return error.DataLengthMismatch;
        try ensureFiniteSlice(src[0..self.dim]);
        var d: usize = 0;
        while (d < self.dim) : (d += 1) self.data[(b * self.dim + d) * 2 + self.offset] = src[d];
    }
};

fn latentToBlockedTensor(state: *const RSFLatentState, blocked: *Tensor) !void {
    const shape = try latentShapeOf(state);
    const dim2 = try checkedMul(shape.dim, 2);
    if (blocked.shape.dims.len != 2) return error.ShapeMismatch;
    if (blocked.shape.dims[0] != shape.batch or blocked.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (blocked.data.len < try checkedMul(shape.batch, dim2)) return error.DataLengthMismatch;
    var b: usize = 0;
    while (b < shape.batch) : (b += 1) {
        const dst = blocked.data[b * dim2 .. b * dim2 + dim2];
        var d: usize = 0;
        while (d < shape.dim) : (d += 1) {
            const src = (b * shape.dim + d) * 2;
            dst[d] = state.data.data[src];
            dst[shape.dim + d] = state.data.data[src + 1];
        }
    }
}

fn blockedTensorToLatent(blocked: *const Tensor, state: *RSFLatentState) !void {
    const shape = try latentShapeOf(state);
    const dim2 = try checkedMul(shape.dim, 2);
    if (blocked.shape.dims.len != 2) return error.ShapeMismatch;
    if (blocked.shape.dims[0] != shape.batch or blocked.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (blocked.data.len < try checkedMul(shape.batch, dim2)) return error.DataLengthMismatch;
    try ensureFiniteSlice(blocked.data[0 .. shape.batch * dim2]);
    var b: usize = 0;
    while (b < shape.batch) : (b += 1) {
        const src = blocked.data[b * dim2 .. b * dim2 + dim2];
        var d: usize = 0;
        while (d < shape.dim) : (d += 1) {
            const dst = (b * shape.dim + d) * 2;
            state.data.data[dst] = src[d];
            state.data.data[dst + 1] = src[shape.dim + d];
        }
    }
}

fn allocBlockedLatentTensor(allocator: Allocator, state: *const RSFLatentState) !Tensor {
    const shape = try latentShapeOf(state);
    return Tensor.init(allocator, &[_]usize{ shape.batch, try checkedMul(shape.dim, 2) });
}

fn meanLogDet(per_row: []const f32, batch: usize) !f32 {
    if (per_row.len < batch or batch == 0) return error.DataLengthMismatch;
    var sum: f64 = 0.0;
    for (per_row[0..batch]) |v| sum += @as(f64, @floatCast(v));
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(batch)));
    if (!std.math.isFinite(mean)) return error.NonFinite;
    return mean;
}

pub const RSFLatentState = struct {
    data: Tensor,
    log_det: f32,
    binding: types.RSFBinding,

    pub fn init(allocator: Allocator, model: *const RSF, batch: usize) !RSFLatentState {
        if (batch == 0) return error.InvalidBatchSize;
        const dim = try model.dim();
        const model_id = try handleId(model.id);
        var data = try Tensor.init(allocator, &[_]usize{ batch, dim, 2 });
        errdefer data.deinit();
        return .{ .data = data, .log_det = 0.0, .binding = types.RSFBinding.model(.latent_state, model_id, dim) };
    }

    pub fn fromHalves(allocator: Allocator, model: *const RSF, x1: *const Tensor, x2: *const Tensor) !RSFLatentState {
        try validateTensor2D(x1);
        try validateTensor2D(x2);
        const dim = try model.dim();
        if (x1.shape.dims[1] != dim or x2.shape.dims[1] != dim) return error.ShapeMismatch;
        if (x1.shape.dims[0] != x2.shape.dims[0]) return error.ShapeMismatch;
        const batch = x1.shape.dims[0];
        try ensureFiniteSlice(x1.data);
        try ensureFiniteSlice(x2.data);
        var state = try init(allocator, model, batch);
        errdefer state.deinit();
        const even = try state.evenView();
        const odd = try state.oddView();
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            try even.copyRowFrom(b, x1.data[b * dim .. b * dim + dim]);
            try odd.copyRowFrom(b, x2.data[b * dim .. b * dim + dim]);
        }
        return state;
    }

    pub fn deinit(self: *RSFLatentState) void {
        self.data.deinit();
        self.log_det = 0.0;
    }

    pub fn clone(self: *const RSFLatentState, allocator: Allocator) !RSFLatentState {
        const shape = try latentShapeOf(self);
        var data = try Tensor.init(allocator, &[_]usize{ shape.batch, shape.dim, 2 });
        errdefer data.deinit();
        try ensureFiniteSlice(self.data.data);
        @memcpy(data.data, self.data.data);
        return .{ .data = data, .log_det = self.log_det, .binding = self.binding };
    }

    pub fn evenView(self: *RSFLatentState) !LatentHalfView {
        const shape = try latentShapeOf(self);
        return .{ .data = self.data.data, .batch = shape.batch, .dim = shape.dim, .offset = 0 };
    }

    pub fn oddView(self: *RSFLatentState) !LatentHalfView {
        const shape = try latentShapeOf(self);
        return .{ .data = self.data.data, .batch = shape.batch, .dim = shape.dim, .offset = 1 };
    }

    pub fn requireModel(self: *const RSFLatentState, model: *const RSF) !void {
        const model_id = try handleId(model.id);
        const dim = try model.dim();
        try self.binding.requireSpace(.latent_state);
        try self.binding.requireModel(model_id);
        try self.binding.requireDim(dim);
        const shape = try latentShapeOf(self);
        if (shape.dim != dim) return types.RSFBindingError.RSFDimMismatch;
    }

    pub fn blockedLength(self: *const RSFLatentState) !usize {
        const shape = try latentShapeOf(self);
        return try checkedMul(shape.batch, try checkedMul(shape.dim, 2));
    }

    pub fn forwardThrough(self: *RSFLatentState, model: *RSF) !void {
        self.log_det += try model.forwardLatentWithLogDet(self);
    }

    pub fn inverseThrough(self: *RSFLatentState, model: *RSF) !void {
        self.log_det -= try model.inverseLatentWithLogDet(self);
    }

    pub fn roundtripError(self: *const RSFLatentState, model: *RSF, allocator: Allocator) !f32 {
        var working = try self.clone(allocator);
        defer working.deinit();
        try working.forwardThrough(model);
        try working.inverseThrough(model);
        const total = try self.blockedLength();
        var num: f64 = 0.0;
        var den: f64 = 0.0;
        var k: usize = 0;
        while (k < total) : (k += 1) {
            const original: f64 = @floatCast(self.data.data[k]);
            const recovered: f64 = @floatCast(working.data.data[k]);
            const diff = recovered - original;
            num += diff * diff;
            den += original * original;
        }
        if (den <= 0.0) return @as(f32, @floatCast(@sqrt(num)));
        return @as(f32, @floatCast(@sqrt(num / den)));
    }
};

pub const RSF = struct {
    id: u64 = 0,
    ctrl: ?*RSFCore = null,
    pub fn init(allocator: Allocator, model_dim: usize, num_layers: usize) !RSF {
        return initWithConfig(allocator, model_dim, num_layers, .{});
    }
    pub fn initWithConfig(allocator: Allocator, model_dim: usize, num_layers: usize, cfg: RSFConfig) !RSF {
        try validateModelConfigValues(model_dim, num_layers, cfg);
        _ = try checkedMul(model_dim, coupling_width);
        _ = try checkedMul(model_dim, 2);
        const core = try allocator.create(RSFCore);
        errdefer allocator.destroy(core);
        core.* = .{
            .allocator = allocator,
            .dim = model_dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(LayerCore, num_layers),
            .cfg = cfg,
            .rwlock = .{},
            .gpu_accel = null,
            .gpu_available = std.atomic.Value(u8).init(0),
            .gpu_weight_version = 0,
            .cpu_weight_version = 1,
            .f16_buf = null,
            .oftb = try initOFTBForConfig(model_dim, cfg.global_diffusion),
        };
        errdefer {
            if (core.gpu_accel) |*ga| {
                ga.deinit();
                core.gpu_accel = null;
            }
            if (core.f16_buf) |buf| {
                allocator.free(buf);
                core.f16_buf = null;
            }
            core.gpu_available.store(0, .monotonic);
            core.oftb.deinit();
        }
        errdefer allocator.free(core.layers);
        var initialized: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized) : (j += 1) core.layers[j].deinitOwned();
        }
        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            const seed_base = try checkedMulU64(@as(u64, @intCast(l)), 10007);
            const layer_cfg = RSFLayerConfig{
                .clip_min = cfg.clip_min,
                .clip_max = cfg.clip_max,
                .seed_offset = seed_base,
                .grad_mean = cfg.grad_mean,
            };
            core.layers[l] = try LayerCore.initOwned(allocator, model_dim, layer_cfg);
            initialized += 1;
        }
        try validateModelMetadata(core);
        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch disableGPU(core);
        }
        const id = try registerModelCore(core);
        assignLayerBindings(core, id);
        return RSF{ .id = id, .ctrl = core };
    }
    pub fn deinit(self: *RSF) void {
        const id = self.id;
        if (id == 0) return;
        self.id = 0;
        self.ctrl = null;
        requestDestroyModelCore(id);
    }
    pub fn isGPUAvailable(self: *const RSF) bool {
        const id = handleId(self.id) catch return false;
        const core = acquireModelCore(id) catch return false;
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        return modelGPUCompatible(core) and core.gpu_available.load(.monotonic) != 0 and core.gpu_weight_version == core.cpu_weight_version and core.gpu_accel != null;
    }
    pub fn syncWeightsToGPU(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try syncAllLayersGPU(core);
    }
    pub fn ensureGradients(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        var i: usize = 0;
        while (i < layer_count) : (i += 1) try core.layers[i].ensureGradients();
    }
    pub fn zeroGradients(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        for (core.layers) |*layer| layer.zeroGradients();
    }
    pub fn gradientL2Norm(self: *const RSF) !f32 {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const layer_count = try checkedModelLayerCount(core);
        var total: f64 = 0.0;
        var i: usize = 0;
        while (i < layer_count) : (i += 1) {
            total += try core.layers[i].gradientSquaredSum();
        }
        const result: f32 = @floatCast(@sqrt(total));
        if (!std.math.isFinite(result)) return error.NonFinite;
        return result;
    }
    pub fn applyGradientStep(self: *RSF, learning_rate: f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        bumpWeightVersion(core);
        core.gpu_available.store(0, .monotonic);
        defer refreshGPUAfterWeightChange(core);
        var i: usize = 0;
        while (i < layer_count) : (i += 1) {
            try core.layers[i].applyGradientStep(learning_rate);
        }
        try validateModelMetadata(core);
    }
    pub fn forwardCPU(self: *RSF, x: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardOnCore(core, x);
    }
    pub fn forward(self: *RSF, x: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        try validateTensor2D(x);
        const dim2 = try checkedMul(core.dim, 2);
        if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
        if (x.shape.dims[0] == 0) return error.InvalidBatchSize;
        if (gpuPathSelected(core)) {
            core.rwlock.lock();
            defer core.rwlock.unlock();
            try forwardDispatchOnCore(core, x);
        } else {
            core.rwlock.lockShared();
            defer core.rwlock.unlockShared();
            try forwardOnCore(core, x);
        }
    }
    pub fn inverse(self: *RSF, y: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseOnCore(core, y);
    }
    pub fn backward(self: *RSF, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try backwardOnCore(core, grad_output, input, output, grad_input_out, 0.0);
    }
    pub fn forwardWithLogDet(self: *RSF, x: *Tensor, logdet_per_row: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardLogDetOnCore(core, x, logdet_per_row);
    }
    pub fn meanLogDetJacobian(self: *RSF, x: *const Tensor) !f32 {
        try validateTensor2D(x);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const batch_size = x.shape.dims[0];
        if (batch_size == 0) return error.InvalidBatchSize;
        const allocator = scratchAllocator();
        var probe = try tensorClone(allocator, x);
        defer probe.deinit();
        const logdet_per_row = try allocator.alloc(f32, batch_size);
        defer allocator.free(logdet_per_row);
        try forwardLogDetOnCore(core, &probe, logdet_per_row);
        var total: f32 = 0.0;
        var b: usize = 0;
        while (b < batch_size) : (b += 1) total += logdet_per_row[b];
        return total / @as(f32, @floatFromInt(batch_size));
    }
    pub fn backwardWithLogDet(self: *RSF, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor, logdet_weight: f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try backwardOnCore(core, grad_output, input, output, grad_input_out, logdet_weight);
    }
    pub fn notifyWeightsChanged(self: *RSF) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        bumpWeightVersion(core);
        core.gpu_available.store(0, .monotonic);
        try validateModelMetadata(core);
        refreshGPUAfterWeightChange(core);
    }
    pub fn verifyInvertible(self: *RSF, x: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const allocator = scratchAllocator();
        var y = try tensorClone(allocator, x);
        defer y.deinit();
        if (gpuPathSelected(core)) {
            core.rwlock.lock();
            defer core.rwlock.unlock();
            try forwardDispatchOnCore(core, &y);
            try inverseOnCore(core, &y);
        } else {
            core.rwlock.lockShared();
            defer core.rwlock.unlockShared();
            try forwardOnCore(core, &y);
            try inverseOnCore(core, &y);
        }
        return tensorAllCloseEq(x, &y, abs_tol, rel_tol);
    }
    pub fn dim(self: *const RSF) !usize {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return core.dim;
    }

    pub fn layerCount(self: *const RSF) !usize {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return try checkedModelLayerCount(core);
    }

    pub fn globalDiffusionEnabled(self: *const RSF) !bool {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return core.cfg.global_diffusion and core.oftb.diffusionEnabled();
    }

    pub fn diffusionLayout(self: *const RSF) !types.RSFDiffusionLayout {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return core.oftb.diffusionLayout() orelse error.DiffusionDisabled;
    }

    pub fn latentBinding(self: *const RSF) !types.RSFBinding {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        return types.RSFBinding.model(.latent_state, id, core.dim);
    }

    pub fn layerBindingsFor(self: *RSF, layer: usize) !LayerBindings {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const target = try checkedLayerIndex(core, layer);
        if (target.model_id != id or target.layer_index != layer) {
            core.rwlock.lock();
            defer core.rwlock.unlock();
            target.model_id = id;
            target.layer_index = layer;
        }
        return layerBindings(id, layer, core.dim);
    }

    pub fn readLayerWeights(self: *const RSF, layer: usize, s_out: []f32, t_out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_out.len < expected or t_out.len < expected) return error.DataLengthMismatch;
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        try validateTensor2DShape(&target.s_weight, core.dim, coupling_width);
        try validateTensor2DShape(&target.t_weight, core.dim, coupling_width);
        try ensureFiniteSlice(target.s_weight.data[0..expected]);
        try ensureFiniteSlice(target.t_weight.data[0..expected]);
        @memcpy(s_out[0..expected], target.s_weight.data[0..expected]);
        @memcpy(t_out[0..expected], target.t_weight.data[0..expected]);
    }

    pub fn writeLayerWeights(self: *RSF, layer: usize, s_in: []const f32, t_in: []const f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_in.len < expected or t_in.len < expected) return error.DataLengthMismatch;
        try ensureFiniteSlice(s_in[0..expected]);
        try ensureFiniteSlice(t_in[0..expected]);
        target.rwlock.lock();
        defer target.rwlock.unlock();
        try validateTensor2DShape(&target.s_weight, core.dim, coupling_width);
        try validateTensor2DShape(&target.t_weight, core.dim, coupling_width);
        @memcpy(target.s_weight.data[0..expected], s_in[0..expected]);
        @memcpy(target.t_weight.data[0..expected], t_in[0..expected]);
        bumpWeightVersion(core);
        core.gpu_available.store(0, .monotonic);
        try validateModelMetadata(core);
        refreshGPUAfterWeightChange(core);
    }

    pub fn readLayerGradients(self: *const RSF, layer: usize, s_out: []f32, t_out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_out.len < expected or t_out.len < expected) return error.DataLengthMismatch;
        const s_grad = target.s_weight_grad orelse return error.NoGradients;
        const t_grad = target.t_weight_grad orelse return error.NoGradients;
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        try validateTensor2DShape(&s_grad, core.dim, coupling_width);
        try validateTensor2DShape(&t_grad, core.dim, coupling_width);
        try ensureFiniteSlice(s_grad.data[0..expected]);
        try ensureFiniteSlice(t_grad.data[0..expected]);
        @memcpy(s_out[0..expected], s_grad.data[0..expected]);
        @memcpy(t_out[0..expected], t_grad.data[0..expected]);
    }

    pub fn readLayerGradientProducts(self: *const RSF, layer: usize, branch: RSFBranch, out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const model_dim = core.dim;
        const expected = try checkedMul(model_dim, 3);
        if (out.len < expected) return error.InvalidDataLength;
        const grad = switch (branch) {
            .s => target.s_weight_grad orelse return error.NoGradients,
            .t => target.t_weight_grad orelse return error.NoGradients,
        };
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        try validateTensor2DShape(&grad, model_dim, coupling_width);
        try ensureFiniteSlice(grad.data);
        var d: usize = 0;
        while (d < model_dim) : (d += 1) {
            const g_w = grad.data[d * coupling_width + WEIGHT_COLUMN];
            const g_b = grad.data[d * coupling_width + BIAS_COLUMN];
            out[d * 3 + 0] = g_w * g_w;
            out[d * 3 + 1] = g_w * g_b;
            out[d * 3 + 2] = g_b * g_b;
        }
    }

    pub fn readLayerGaussNewton(self: *const RSF, layer: usize, s_block_out: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, 3);
        if (s_block_out.len != expected) return error.InvalidDataLength;
        target.rwlock.lockShared();
        defer target.rwlock.unlockShared();
        if (target.gn_block) |blk| {
            if (blk.len < expected) return error.InvalidModelState;
            try ensureFiniteSlice(blk[0..expected]);
            @memcpy(s_block_out[0..expected], blk[0..expected]);
            return;
        }
        @memset(s_block_out[0..expected], 0.0);
    }

    pub fn zeroLayerGaussNewton(self: *RSF, layer: usize) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const target = try checkedLayerIndex(core, layer);
        target.rwlock.lock();
        defer target.rwlock.unlock();
        target.zeroGaussNewton();
    }

    pub fn scaleLayerGradients(self: *RSF, scale: f32) !void {
        if (!std.math.isFinite(scale)) return error.NonFinite;
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const layer_count = try checkedModelLayerCount(core);
        var i: usize = 0;
        while (i < layer_count) : (i += 1) {
            const target = &core.layers[i];
            try target.ensureGradients();
            const s_grad = target.s_weight_grad orelse return error.NoGradients;
            const t_grad = target.t_weight_grad orelse return error.NoGradients;
            try ensureFiniteSlice(s_grad.data);
            try ensureFiniteSlice(t_grad.data);
            for (s_grad.data) |v| {
                if (!std.math.isFinite(v * scale)) return error.NonFinite;
            }
            for (t_grad.data) |v| {
                if (!std.math.isFinite(v * scale)) return error.NonFinite;
            }
            target.rwlock.lock();
            defer target.rwlock.unlock();
            for (s_grad.data) |*v| v.* *= scale;
            for (t_grad.data) |*v| v.* *= scale;
        }
    }

    pub fn accumulateLayerGradients(self: *RSF, layer: usize, s_in: []const f32, t_in: []const f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const target = try checkedLayerIndex(core, layer);
        const expected = try checkedMul(core.dim, coupling_width);
        if (s_in.len < expected or t_in.len < expected) return error.DataLengthMismatch;
        try ensureFiniteSlice(s_in[0..expected]);
        try ensureFiniteSlice(t_in[0..expected]);
        try target.ensureGradients();
        const s_grad = target.s_weight_grad orelse return error.NoGradients;
        const t_grad = target.t_weight_grad orelse return error.NoGradients;
        try validateTensor2DShape(&s_grad, core.dim, coupling_width);
        try validateTensor2DShape(&t_grad, core.dim, coupling_width);
        try ensureFiniteSlice(s_grad.data);
        try ensureFiniteSlice(t_grad.data);
        var k: usize = 0;
        while (k < expected) : (k += 1) {
            if (!std.math.isFinite(s_grad.data[k] + s_in[k])) return error.NonFinite;
            if (!std.math.isFinite(t_grad.data[k] + t_in[k])) return error.NonFinite;
        }
        target.rwlock.lock();
        defer target.rwlock.unlock();
        k = 0;
        while (k < expected) : (k += 1) {
            s_grad.data[k] += s_in[k];
            t_grad.data[k] += t_in[k];
        }
    }

    pub fn inverseWithLogDet(self: *RSF, y: *Tensor, logdet_per_row: []f32) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseLogDetOnCore(core, y, logdet_per_row);
    }

    pub fn forwardLatentWithLogDet(self: *RSF, state: *RSFLatentState) !f32 {
        try state.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const shape = try latentShapeOf(state);
        const allocator = scratchAllocator();
        var blocked = try allocBlockedLatentTensor(allocator, state);
        defer blocked.deinit();
        try latentToBlockedTensor(state, &blocked);
        const per_row = try allocator.alloc(f32, shape.batch);
        defer allocator.free(per_row);
        core.rwlock.lockShared();
        try forwardLogDetOnCore(core, &blocked, per_row);
        core.rwlock.unlockShared();
        try blockedTensorToLatent(&blocked, state);
        return try meanLogDet(per_row, shape.batch);
    }

    pub fn inverseLatentWithLogDet(self: *RSF, state: *RSFLatentState) !f32 {
        try state.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const shape = try latentShapeOf(state);
        const allocator = scratchAllocator();
        var blocked = try allocBlockedLatentTensor(allocator, state);
        defer blocked.deinit();
        try latentToBlockedTensor(state, &blocked);
        const per_row = try allocator.alloc(f32, shape.batch);
        defer allocator.free(per_row);
        core.rwlock.lockShared();
        try inverseLogDetOnCore(core, &blocked, per_row);
        core.rwlock.unlockShared();
        try blockedTensorToLatent(&blocked, state);
        return try meanLogDet(per_row, shape.batch);
    }

    pub fn forwardLatent(self: *RSF, state: *RSFLatentState) !void {
        _ = try self.forwardLatentWithLogDet(state);
    }

    pub fn inverseLatent(self: *RSF, state: *RSFLatentState) !void {
        _ = try self.inverseLatentWithLogDet(state);
    }

    pub fn backwardLatent(self: *RSF, grad_output: *const RSFLatentState, input: *const RSFLatentState, output: *const RSFLatentState, grad_input_out: *RSFLatentState, logdet_weight: f32) !void {
        try grad_output.requireModel(self);
        try input.requireModel(self);
        try output.requireModel(self);
        try grad_input_out.requireModel(self);
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        const allocator = scratchAllocator();
        var g_out = try allocBlockedLatentTensor(allocator, grad_output);
        defer g_out.deinit();
        var x_in = try allocBlockedLatentTensor(allocator, input);
        defer x_in.deinit();
        var y_out = try allocBlockedLatentTensor(allocator, output);
        defer y_out.deinit();
        var dx = try allocBlockedLatentTensor(allocator, grad_input_out);
        defer dx.deinit();
        try latentToBlockedTensor(grad_output, &g_out);
        try latentToBlockedTensor(input, &x_in);
        try latentToBlockedTensor(output, &y_out);
        try backwardOnCore(core, &g_out, &x_in, &y_out, &dx, logdet_weight);
        try blockedTensorToLatent(&dx, grad_input_out);
    }

    pub fn save(self: *const RSF, path: []const u8) !void {
        const id = try handleId(self.id);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        const allocator = scratchAllocator();
        core.rwlock.lockShared();
        var snapshot = snapshotModelForSave(allocator, core) catch |err| {
            core.rwlock.unlockShared();
            return err;
        };
        core.rwlock.unlockShared();
        defer snapshot.deinit();
        try writeSnapshotToPath(&snapshot, path, allocator);
    }
    pub fn load(allocator: Allocator, path: []const u8) !RSF {
        return loadWithConfig(allocator, path, null);
    }
    pub fn loadWithConfig(allocator: Allocator, path: []const u8, policy: ?RSFConfig) !RSF {
        if (policy) |p| {
            try validateClipRange(p.clip_min, p.clip_max);
            if (p.max_dim == 0 or p.max_layers == 0) return error.InvalidConfig;
        }
        const file = try core_io.openFilePath(path, .{});
        defer file.close();
        var buffered = std.io.bufferedReader(file.reader());
        const r = buffered.reader();
        var magic: [4]u8 = undefined;
        try r.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "RSF0")) return error.BadFileFormat;
        const version = try r.readInt(u32, .little);
        if (version != SAVE_VERSION and version != SAVE_VERSION_LEGACY) return error.UnsupportedVersion;
        const num_layers_u64 = try r.readInt(u64, .little);
        const dim_u64 = try r.readInt(u64, .little);
        if (num_layers_u64 == 0) return error.InvalidLayerCount;
        if (dim_u64 == 0) return error.InvalidDimension;
        const policy_max_dim: usize = if (policy) |p| p.max_dim else (1 << 20);
        const policy_max_layers: usize = if (policy) |p| p.max_layers else (1 << 20);
        if (num_layers_u64 > @as(u64, @intCast(policy_max_layers)) or dim_u64 > @as(u64, @intCast(policy_max_dim))) return error.TooLarge;
        const num_layers = try checkedCastU64ToUsize(num_layers_u64);
        const model_dim = try checkedCastU64ToUsize(dim_u64);
        _ = try checkedMul(model_dim, coupling_width);
        _ = try checkedMul(model_dim, 2);
        var hasher = std.hash.Crc32.init();
        hasher.update("RSF0");
        crcUpdateU32LE(&hasher, version);
        crcUpdateU64LE(&hasher, num_layers_u64);
        crcUpdateU64LE(&hasher, dim_u64);
        const clip_min_bits = try r.readInt(u32, .little);
        const clip_max_bits = try r.readInt(u32, .little);
        const clip_min: f32 = @bitCast(clip_min_bits);
        const clip_max: f32 = @bitCast(clip_max_bits);
        const grad_mean = try readEncodedBool(r);
        try validateClipRange(clip_min, clip_max);
        if (policy) |p| {
            if (clip_min != p.clip_min or clip_max != p.clip_max or grad_mean != p.grad_mean) return error.PolicyMismatch;
        }
        crcUpdateU32LE(&hasher, clip_min_bits);
        crcUpdateU32LE(&hasher, clip_max_bits);
        crcUpdateU8(&hasher, if (grad_mean) @as(u8, 1) else @as(u8, 0));
        const saved_max_dim_u64 = try r.readInt(u64, .little);
        const saved_max_layers_u64 = try r.readInt(u64, .little);
        crcUpdateU64LE(&hasher, saved_max_dim_u64);
        crcUpdateU64LE(&hasher, saved_max_layers_u64);
        if (saved_max_dim_u64 == 0 or saved_max_layers_u64 == 0) return error.InvalidConfig;
        if (saved_max_dim_u64 < dim_u64 or saved_max_layers_u64 < num_layers_u64) return error.InvalidConfig;
        const effective_max_dim: usize = if (policy) |p| p.max_dim else try checkedCastU64ToUsize(saved_max_dim_u64);
        const effective_max_layers: usize = if (policy) |p| p.max_layers else try checkedCastU64ToUsize(saved_max_layers_u64);
        var global_diffusion = false;
        if (version == SAVE_VERSION) {
            global_diffusion = try readEncodedBool(r);
            crcUpdateU8(&hasher, if (global_diffusion) @as(u8, 1) else @as(u8, 0));
        }
        if (policy) |p| {
            if (p.global_diffusion != global_diffusion) return error.PolicyMismatch;
        }
        const loaded_cfg = RSFConfig{
            .clip_min = clip_min,
            .clip_max = clip_max,
            .grad_mean = grad_mean,
            .max_dim = effective_max_dim,
            .max_layers = effective_max_layers,
            .global_diffusion = global_diffusion,
        };
        try validateModelConfigValues(model_dim, num_layers, loaded_cfg);
        const core = try allocator.create(RSFCore);
        errdefer allocator.destroy(core);
        core.* = .{
            .allocator = allocator,
            .dim = model_dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(LayerCore, num_layers),
            .cfg = loaded_cfg,
            .rwlock = .{},
            .gpu_accel = null,
            .gpu_available = std.atomic.Value(u8).init(0),
            .gpu_weight_version = 0,
            .cpu_weight_version = 1,
            .f16_buf = null,
            .oftb = try initOFTBForConfig(model_dim, loaded_cfg.global_diffusion),
        };
        errdefer {
            if (core.gpu_accel) |*ga| {
                ga.deinit();
                core.gpu_accel = null;
            }
            if (core.f16_buf) |buf| {
                allocator.free(buf);
                core.f16_buf = null;
            }
            core.gpu_available.store(0, .monotonic);
            core.oftb.deinit();
        }
        errdefer allocator.free(core.layers);
        var initialized: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized) : (j += 1) core.layers[j].deinitOwned();
        }
        var i: usize = 0;
        while (i < num_layers) : (i += 1) {
            const layer_clip_min_bits = try r.readInt(u32, .little);
            const layer_clip_max_bits = try r.readInt(u32, .little);
            const layer_clip_min: f32 = @bitCast(layer_clip_min_bits);
            const layer_clip_max: f32 = @bitCast(layer_clip_max_bits);
            const layer_grad_mean = try readEncodedBool(r);
            try validateClipRange(layer_clip_min, layer_clip_max);
            if (layer_clip_min != clip_min or layer_clip_max != clip_max or layer_grad_mean != grad_mean) return error.InvalidConfig;
            crcUpdateU32LE(&hasher, layer_clip_min_bits);
            crcUpdateU32LE(&hasher, layer_clip_max_bits);
            crcUpdateU8(&hasher, if (layer_grad_mean) @as(u8, 1) else @as(u8, 0));
            var s_w_new = try readTensorData(allocator, r, model_dim, coupling_width);
            errdefer s_w_new.deinit();
            var t_w_new = try readTensorData(allocator, r, model_dim, coupling_width);
            errdefer t_w_new.deinit();
            try validateTensor2DShape(&s_w_new, model_dim, coupling_width);
            try validateTensor2DShape(&t_w_new, model_dim, coupling_width);
            try ensureFiniteSlice(s_w_new.data);
            try ensureFiniteSlice(t_w_new.data);
            hashTensorData(&hasher, &s_w_new);
            hashTensorData(&hasher, &t_w_new);
            core.layers[i] = .{
                .s_weight = s_w_new,
                .t_weight = t_w_new,
                .s_weight_grad = null,
                .t_weight_grad = null,
                .dim = model_dim,
                .allocator = allocator,
                .clip_min = layer_clip_min,
                .clip_max = layer_clip_max,
                .grad_mean = layer_grad_mean,
                .gn_block = null,
                .model_id = 0,
                .layer_index = i,
                .rwlock = .{},
            };
            initialized += 1;
        }
        if (version == SAVE_VERSION) {
            const gn_len = try checkedMul(model_dim, 3);
            var g: usize = 0;
            while (g < num_layers) : (g += 1) {
                const blk = try allocator.alloc(f32, gn_len);
                errdefer if (core.layers[g].gn_block == null) allocator.free(blk);
                var k: usize = 0;
                while (k < gn_len) : (k += 1) {
                    const bits = try r.readInt(u32, .little);
                    crcUpdateU32LE(&hasher, bits);
                    const value: f32 = @bitCast(bits);
                    if (!std.math.isFinite(value) or value < 0.0) return error.NonFinite;
                    blk[k] = value;
                }
                core.layers[g].gn_block = blk;
            }
        }
        const stored_crc = try r.readInt(u32, .little);
        if (stored_crc != hasher.final()) return error.ChecksumMismatch;
        var eof_buf: [1]u8 = undefined;
        if ((try r.read(&eof_buf)) != 0) return error.TrailingData;
        try validateModelMetadata(core);
        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch disableGPU(core);
        }
        const id = try registerModelCore(core);
        assignLayerBindings(core, id);
        return RSF{ .id = id, .ctrl = core };
    }
    pub fn saveLoadRoundtrip(allocator: Allocator, self: *const RSF, path: []const u8, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        try self.save(path);
        var loaded = try RSF.load(allocator, path);
        defer loaded.deinit();
        const id1 = try handleId(self.id);
        const core1 = try acquireModelCore(id1);
        defer releaseModelCore(id1);
        const id2 = try handleId(loaded.id);
        const core2 = try acquireModelCore(id2);
        defer releaseModelCore(id2);
        core1.rwlock.lockShared();
        defer core1.rwlock.unlockShared();
        core2.rwlock.lockShared();
        defer core2.rwlock.unlockShared();
        const layer_count1 = try checkedModelLayerCount(core1);
        const layer_count2 = try checkedModelLayerCount(core2);
        if (core1.dim != core2.dim) return false;
        if (layer_count1 != layer_count2) return false;
        if (core1.cfg.clip_min != core2.cfg.clip_min or core1.cfg.clip_max != core2.cfg.clip_max or core1.cfg.grad_mean != core2.cfg.grad_mean) return false;
        if (core1.cfg.max_dim != core2.cfg.max_dim or core1.cfg.max_layers != core2.cfg.max_layers) return false;
        var i: usize = 0;
        while (i < layer_count1) : (i += 1) {
            if (!(try tensorAllCloseEq(&core1.layers[i].s_weight, &core2.layers[i].s_weight, abs_tol, rel_tol))) return false;
            if (!(try tensorAllCloseEq(&core1.layers[i].t_weight, &core2.layers[i].t_weight, abs_tol, rel_tol))) return false;
            if (core1.layers[i].clip_min != core2.layers[i].clip_min or core1.layers[i].clip_max != core2.layers[i].clip_max or core1.layers[i].grad_mean != core2.layers[i].grad_mean) return false;
        }
        return true;
    }
};
fn crcUpdateU32LE(hasher: *std.hash.Crc32, v: u32) void {
    const le = std.mem.nativeToLittle(u32, v);
    hasher.update(std.mem.asBytes(&le));
}
fn crcUpdateU64LE(hasher: *std.hash.Crc32, v: u64) void {
    const le = std.mem.nativeToLittle(u64, v);
    hasher.update(std.mem.asBytes(&le));
}
fn crcUpdateU8(hasher: *std.hash.Crc32, v: u8) void {
    hasher.update(&.{v});
}
fn writeTensorData(w: anytype, hasher: *std.hash.Crc32, t: *const Tensor) !void {
    try validateTensor2D(t);
    try ensureFiniteSlice(t.data);
    const rows = t.shape.dims[0];
    const cols = t.shape.dims[1];
    try w.writeInt(u64, TENSOR_RANK_TAG, .little);
    crcUpdateU64LE(hasher, TENSOR_RANK_TAG);
    try w.writeInt(u64, @intCast(rows), .little);
    try w.writeInt(u64, @intCast(cols), .little);
    crcUpdateU64LE(hasher, @intCast(rows));
    crcUpdateU64LE(hasher, @intCast(cols));
    for (t.data) |v| {
        const bits = @as(u32, @bitCast(v));
        try w.writeInt(u32, bits, .little);
        crcUpdateU32LE(hasher, bits);
    }
}
fn hashTensorData(hasher: *std.hash.Crc32, t: *const Tensor) void {
    crcUpdateU64LE(hasher, TENSOR_RANK_TAG);
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[0]));
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[1]));
    for (t.data) |v| crcUpdateU32LE(hasher, @as(u32, @bitCast(v)));
}
fn readEncodedBool(r: anytype) !bool {
    const b = try r.readByte();
    return switch (b) {
        0 => false,
        1 => true,
        else => error.BadFileFormat,
    };
}
fn readTensorData(allocator: Allocator, r: anytype, expected_rows: usize, expected_cols: usize) !Tensor {
    if ((try r.readInt(u64, .little)) != TENSOR_RANK_TAG) return error.BadFileFormat;
    const d0_u64 = try r.readInt(u64, .little);
    const d1_u64 = try r.readInt(u64, .little);
    const expected_rows_u64: u64 = @intCast(expected_rows);
    const expected_cols_u64: u64 = @intCast(expected_cols);
    if (d0_u64 != expected_rows_u64 or d1_u64 != expected_cols_u64) return error.ShapeMismatch;
    const expected = try checkedMul(expected_rows, expected_cols);
    var t = try Tensor.init(allocator, &.{ expected_rows, expected_cols });
    errdefer t.deinit();
    var i: usize = 0;
    while (i < expected) : (i += 1) {
        const bits = try r.readInt(u32, .little);
        t.data[i] = @as(f32, @bitCast(bits));
    }
    return t;
}
const TempFile = struct {
    file: std.fs.File,
    tmp_name: []u8,
};
fn hexEncodeLower(dst: []u8, src: []const u8) []u8 {
    const alphabet = "0123456789abcdef";
    var i: usize = 0;
    while (i < src.len) : (i += 1) {
        const hi: usize = @intCast((src[i] >> 4) & 0x0f);
        const lo: usize = @intCast(src[i] & 0x0f);
        dst[i * 2] = alphabet[hi];
        dst[i * 2 + 1] = alphabet[lo];
    }
    return dst[0 .. src.len * 2];
}
fn createUniqueTempFile(dir: *std.fs.Dir, allocator: Allocator, base_name: []const u8) !TempFile {
    var attempt: usize = 0;
    while (attempt < 64) : (attempt += 1) {
        var rnd: [16]u8 = undefined;
        std.crypto.random.bytes(&rnd);
        var hex_buf: [32]u8 = undefined;
        const hex = hexEncodeLower(&hex_buf, &rnd);
        const tmp_name = try std.fmt.allocPrint(allocator, ".{s}.tmp.{s}", .{ base_name, hex });
        errdefer allocator.free(tmp_name);
        const file = dir.createFile(tmp_name, .{ .exclusive = true, .mode = 0o600 }) catch |e| switch (e) {
            error.PathAlreadyExists => {
                allocator.free(tmp_name);
                continue;
            },
            else => return e,
        };
        return .{ .file = file, .tmp_name = tmp_name };
    }
    return error.TempFileCollision;
}
fn syncDirectory(dir: std.fs.Dir) !void {
    if (comptime builtin.os.tag == .windows) return;
    std.posix.fsync(dir.fd) catch |err| switch (err) {
        error.AccessDenied => return,
        else => return err,
    };
}
fn writeSnapshotToPath(snapshot: *const SavedModelSnapshot, path: []const u8, allocator: Allocator) !void {
    if (snapshot.num_layers != snapshot.layers.len) return error.InvalidModelState;
    try validateModelConfigValues(snapshot.dim, snapshot.num_layers, snapshot.cfg);
    if (path.len == 0) return error.InvalidPath;
    const parent_path = if (std.fs.path.dirname(path)) |p| p else ".";
    const base_name = std.fs.path.basename(path);
    if (base_name.len == 0) return error.InvalidPath;
    var parent_dir = if (std.fs.path.isAbsolute(parent_path)) try std.fs.openDirAbsolute(parent_path, .{ .iterate = true }) else try std.fs.cwd().openDir(parent_path, .{ .iterate = true });
    defer parent_dir.close();
    const temp = try createUniqueTempFile(&parent_dir, allocator, base_name);
    defer allocator.free(temp.tmp_name);
    const file = temp.file;
    var file_open = true;
    var tmp_exists = true;
    errdefer {
        if (file_open) file.close();
        if (tmp_exists) parent_dir.deleteFile(temp.tmp_name) catch {};
    }
    var buffered = std.io.bufferedWriter(file.writer());
    const w = buffered.writer();
    var hasher = std.hash.Crc32.init();
    try w.writeAll("RSF0");
    hasher.update("RSF0");
    try w.writeInt(u32, SAVE_VERSION, .little);
    crcUpdateU32LE(&hasher, SAVE_VERSION);
    try w.writeInt(u64, @intCast(snapshot.num_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.num_layers));
    try w.writeInt(u64, @intCast(snapshot.dim), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.dim));
    const clip_min_bits = @as(u32, @bitCast(snapshot.cfg.clip_min));
    const clip_max_bits = @as(u32, @bitCast(snapshot.cfg.clip_max));
    try w.writeInt(u32, clip_min_bits, .little);
    try w.writeInt(u32, clip_max_bits, .little);
    crcUpdateU32LE(&hasher, clip_min_bits);
    crcUpdateU32LE(&hasher, clip_max_bits);
    const gm_byte: u8 = if (snapshot.cfg.grad_mean) 1 else 0;
    try w.writeByte(gm_byte);
    crcUpdateU8(&hasher, gm_byte);
    try w.writeInt(u64, @intCast(snapshot.cfg.max_dim), .little);
    try w.writeInt(u64, @intCast(snapshot.cfg.max_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.cfg.max_dim));
    crcUpdateU64LE(&hasher, @intCast(snapshot.cfg.max_layers));
    const gd_byte: u8 = if (snapshot.cfg.global_diffusion) 1 else 0;
    try w.writeByte(gd_byte);
    crcUpdateU8(&hasher, gd_byte);
    var i: usize = 0;
    while (i < snapshot.layers.len) : (i += 1) {
        const layer = &snapshot.layers[i];
        if (layer.clip_min != snapshot.cfg.clip_min or layer.clip_max != snapshot.cfg.clip_max or layer.grad_mean != snapshot.cfg.grad_mean) return error.InvalidConfig;
        const lmin_bits = @as(u32, @bitCast(layer.clip_min));
        const lmax_bits = @as(u32, @bitCast(layer.clip_max));
        try w.writeInt(u32, lmin_bits, .little);
        try w.writeInt(u32, lmax_bits, .little);
        crcUpdateU32LE(&hasher, lmin_bits);
        crcUpdateU32LE(&hasher, lmax_bits);
        const lgm: u8 = if (layer.grad_mean) 1 else 0;
        try w.writeByte(lgm);
        crcUpdateU8(&hasher, lgm);
        try writeTensorData(w, &hasher, &layer.s_weight);
        try writeTensorData(w, &hasher, &layer.t_weight);
    }
    const gn_len = try checkedMul(snapshot.dim, 3);
    var g: usize = 0;
    while (g < snapshot.layers.len) : (g += 1) {
        const gn = snapshot.layers[g].gn_block;
        if (gn.len < gn_len) return error.InvalidModelState;
        var k: usize = 0;
        while (k < gn_len) : (k += 1) {
            const bits = @as(u32, @bitCast(gn[k]));
            try w.writeInt(u32, bits, .little);
            crcUpdateU32LE(&hasher, bits);
        }
    }
    try w.writeInt(u32, hasher.final(), .little);
    try buffered.flush();
    try file.sync();
    file.close();
    file_open = false;
    try parent_dir.rename(temp.tmp_name, base_name);
    tmp_exists = false;
    try syncDirectory(parent_dir);
}
test "RSF forward then inverse returns input within 1e-4 tolerance" {
    const allocator = std.testing.allocator;
    var rsf = try RSFLayer.init(allocator, 32);
    defer rsf.deinit();
    var x1 = try Tensor.randomUniform(allocator, &[_]usize{ 4, 32 }, -1.0, 1.0, 99);
    defer x1.deinit();
    var x2 = try Tensor.randomUniform(allocator, &[_]usize{ 4, 32 }, -1.0, 1.0, 100);
    defer x2.deinit();
    var c1 = try tensorClone(allocator, &x1);
    defer c1.deinit();
    var c2 = try tensorClone(allocator, &x2);
    defer c2.deinit();
    try rsf.forward(&x1, &x2);
    try rsf.inverse(&x1, &x2);
    var idx: usize = 0;
    while (idx < x1.data.len) : (idx += 1) {
        try std.testing.expectApproxEqAbs(x1.data[idx], c1.data[idx], 1e-4);
    }
    idx = 0;
    while (idx < x2.data.len) : (idx += 1) {
        try std.testing.expectApproxEqAbs(x2.data[idx], c2.data[idx], 1e-4);
    }
}
test "RSF with OFTB forward then inverse returns input within 1e-4 tolerance" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const num_layers: usize = 4;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ 2, dim * 2 }, -0.5, 0.5, 123);
    defer input.deinit();
    var original = try tensorClone(allocator, &input);
    defer original.deinit();
    try rsf.forwardCPU(&input);
    try rsf.inverse(&input);
    var idx: usize = 0;
    while (idx < input.data.len) : (idx += 1) {
        try std.testing.expectApproxEqAbs(input.data[idx], original.data[idx], 1e-4);
    }
}
test "RSF forwardWithLogDet accumulates analytic log-determinant per row" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const num_layers: usize = 3;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ 2, dim * 2 }, -0.5, 0.5, 4321);
    defer input.deinit();
    var probe = try tensorClone(allocator, &input);
    defer probe.deinit();
    const logdet_rows = try allocator.alloc(f32, 2);
    defer allocator.free(logdet_rows);
    try rsf.forwardWithLogDet(&probe, logdet_rows);
    var recompute = try tensorClone(allocator, &input);
    defer recompute.deinit();
    try rsf.forwardCPU(&recompute);
    var idx: usize = 0;
    while (idx < probe.data.len) : (idx += 1) {
        try std.testing.expectApproxEqAbs(probe.data[idx], recompute.data[idx], 1e-5);
    }
    idx = 0;
    while (idx < logdet_rows.len) : (idx += 1) {
        try std.testing.expect(std.math.isFinite(logdet_rows[idx]));
    }
}
test "RSF backwardWithLogDet with zero weight matches plain backward" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const num_layers: usize = 2;
    const batch: usize = 2;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.3, 0.3, 555);
    defer input.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forwardCPU(&output);
    var grad_output = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.2, 0.2, 556);
    defer grad_output.deinit();
    var grad_zero_weight = try Tensor.init(allocator, &[_]usize{ batch, dim * 2 });
    defer grad_zero_weight.deinit();
    var grad_plain = try Tensor.init(allocator, &[_]usize{ batch, dim * 2 });
    defer grad_plain.deinit();
    try rsf.zeroGradients();
    try rsf.backwardWithLogDet(&grad_output, &input, &output, &grad_zero_weight, 0.0);
    try rsf.zeroGradients();
    try rsf.backward(&grad_output, &input, &output, &grad_plain);
    var idx: usize = 0;
    while (idx < grad_plain.data.len) : (idx += 1) {
        try std.testing.expectEqual(grad_plain.data[idx], grad_zero_weight.data[idx]);
    }
}
test "RSF backwardWithLogDet shifts S-path adjoint by log-determinant weight" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const num_layers: usize = 2;
    const batch: usize = 2;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.3, 0.3, 777);
    defer input.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forwardCPU(&output);
    var grad_output = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.2, 0.2, 778);
    defer grad_output.deinit();
    var grad_plain = try Tensor.init(allocator, &[_]usize{ batch, dim * 2 });
    defer grad_plain.deinit();
    var grad_logdet = try Tensor.init(allocator, &[_]usize{ batch, dim * 2 });
    defer grad_logdet.deinit();
    try rsf.zeroGradients();
    try rsf.backwardWithLogDet(&grad_output, &input, &output, &grad_logdet, 0.25);
    try rsf.zeroGradients();
    try rsf.backward(&grad_output, &input, &output, &grad_plain);
    var differs = false;
    var idx: usize = 0;
    while (idx < grad_plain.data.len) : (idx += 1) {
        if (!std.math.isFinite(grad_logdet.data[idx])) return error.NonFinite;
        if (@abs(grad_logdet.data[idx] - grad_plain.data[idx]) > 1e-7) differs = true;
    }
    try std.testing.expect(differs);
}
test "RSF backward rejects aliased gradient buffers" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    const batch: usize = 2;
    var rsf = try RSF.init(allocator, dim, 2);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.3, 0.3, 991);
    defer input.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forwardCPU(&output);
    var grad_output = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.2, 0.2, 992);
    defer grad_output.deinit();
    try std.testing.expectError(error.AliasedBuffers, rsf.backward(&grad_output, &input, &output, &grad_output));
}
test "RSF applyGradientStep consumes accumulated gradients" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const num_layers: usize = 2;
    const batch: usize = 3;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.3, 0.3, 2024);
    defer input.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forwardCPU(&output);
    var grad_output = try Tensor.randomUniform(allocator, &[_]usize{ batch, dim * 2 }, -0.2, 0.2, 2025);
    defer grad_output.deinit();
    var grad_input = try Tensor.init(allocator, &[_]usize{ batch, dim * 2 });
    defer grad_input.deinit();
    try rsf.ensureGradients();
    try rsf.zeroGradients();
    const norm_empty = try rsf.gradientL2Norm();
    try std.testing.expectEqual(@as(f32, 0.0), norm_empty);
    try rsf.backward(&grad_output, &input, &output, &grad_input);
    const norm_after_backward = try rsf.gradientL2Norm();
    try std.testing.expect(norm_after_backward > 0.0);
    const core = rsf.ctrl orelse return error.MissingControl;
    var before = try tensorClone(allocator, &core.layers[0].s_weight);
    defer before.deinit();
    const version_before = core.cpu_weight_version;
    try rsf.applyGradientStep(0.05);
    try std.testing.expect(core.cpu_weight_version != version_before);
    var changed = false;
    var idx: usize = 0;
    while (idx < before.data.len) : (idx += 1) {
        if (before.data[idx] != core.layers[0].s_weight.data[idx]) changed = true;
        try std.testing.expect(std.math.isFinite(core.layers[0].s_weight.data[idx]));
    }
    try std.testing.expect(changed);
    try std.testing.expect(try rsf.verifyInvertible(&input, 1e-3, 1e-3));
}
test "RSF save and load round-trip preserves weights" {
    const allocator = std.testing.allocator;
    const dim: usize = 6;
    const num_layers: usize = 3;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_path);
    const file_path = try std.fs.path.join(allocator, &.{ dir_path, "model.rsf" });
    defer allocator.free(file_path);
    try std.testing.expect(try RSF.saveLoadRoundtrip(allocator, &rsf, file_path, 0.0, 0.0));
    var strict = try RSF.loadWithConfig(allocator, file_path, .{});
    defer strict.deinit();
    const strict_core = strict.ctrl orelse return error.MissingControl;
    try std.testing.expectEqual(dim, strict_core.dim);
    try std.testing.expectEqual(num_layers, strict_core.num_layers);
    try std.testing.expectError(error.PolicyMismatch, RSF.loadWithConfig(allocator, file_path, .{ .clip_min = -4.0, .clip_max = 4.0 }));
}
test "RSF exact rank-2 spectral norm replaces the power iteration" {
    const allocator = std.testing.allocator;
    const identity = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), try exactSpectralNormRank2(identity[0..], 2), 1e-6);
    const rank_one = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), try exactSpectralNormRank2(rank_one[0..], 2), 1e-6);
    try std.testing.expectError(error.InvalidDimension, exactSpectralNormRank2(identity[0..], 0));
    try std.testing.expectError(error.DataLengthMismatch, exactSpectralNormRank2(identity[0..1], 2));

    var weight = try Tensor.init(allocator, &[_]usize{ 2, coupling_width });
    defer weight.deinit();
    @memcpy(weight.data, rank_one[0..]);
    try constrainSpectralNorm(&weight, 2, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
    const constrained = try exactSpectralNormRank2(weight.data, 2);
    try std.testing.expectApproxEqAbs(LAYER_TARGET_SPECTRAL_NORM, constrained, 1e-6);
    try std.testing.expect(constrained <= LAYER_TARGET_SPECTRAL_NORM * (1.0 + 1e-6));
    const before_idempotence = weight.data[0];
    try constrainSpectralNorm(&weight, 2, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
    try std.testing.expectEqual(before_idempotence, weight.data[0]);
    try std.testing.expectApproxEqAbs(constrained, try exactSpectralNormRank2(weight.data, 2), 0.0);

    var small = try Tensor.init(allocator, &[_]usize{ 2, coupling_width });
    defer small.deinit();
    const small_data = [_]f32{ 0.1, 0.0, 0.0, 0.1 };
    @memcpy(small.data, small_data[0..]);
    try constrainSpectralNorm(&small, 2, coupling_width, LAYER_TARGET_SPECTRAL_NORM);
    for (0..small.data.len) |k| try std.testing.expectEqual(small_data[k], small.data[k]);

    try std.testing.expectError(error.InvalidConfig, constrainSpectralNorm(&weight, 2, coupling_width, 0.0));
    try std.testing.expectError(error.InvalidConfig, constrainSpectralNorm(&weight, 2, coupling_width, std.math.nan(f32)));
    try std.testing.expectError(error.ShapeMismatch, constrainSpectralNorm(&weight, 1, 4, LAYER_TARGET_SPECTRAL_NORM));
    try std.testing.expectError(error.InvalidDimension, constrainSpectralNorm(&weight, 0, coupling_width, LAYER_TARGET_SPECTRAL_NORM));
    try std.testing.expectError(error.DataLengthMismatch, constrainSpectralNorm(&weight, 9, coupling_width, LAYER_TARGET_SPECTRAL_NORM));
    try std.testing.expectEqual(LAYER_TARGET_SPECTRAL_NORM, layerTargetSpectralNorm());

    var layer = try LayerCore.initOwned(allocator, 8, .{});
    defer layer.deinitOwned();
    const sigma_s = try exactSpectralNormRank2(layer.s_weight.data, 8);
    const sigma_t = try exactSpectralNormRank2(layer.t_weight.data, 8);
    try std.testing.expect(sigma_s <= LAYER_TARGET_SPECTRAL_NORM * (1.0 + 1e-6));
    try std.testing.expect(sigma_t <= LAYER_TARGET_SPECTRAL_NORM * (1.0 + 1e-6));
}

test "RSF backward input gradients match central finite differences" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    const dim2: usize = dim * 2;
    const num_layers: usize = 2;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ 1, dim2 }, -0.4, 0.4, 31337);
    defer input.deinit();
    var seeds = try Tensor.randomUniform(allocator, &[_]usize{ 1, dim2 }, -0.7, 0.7, 31338);
    defer seeds.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forwardCPU(&output);
    var grad_input = try Tensor.init(allocator, &[_]usize{ 1, dim2 });
    defer grad_input.deinit();
    try rsf.zeroGradients();
    try rsf.backward(&seeds, &input, &output, &grad_input);
    const h: f32 = 1.0e-3;
    var k: usize = 0;
    while (k < dim2) : (k += 1) {
        var plus = try tensorClone(allocator, &input);
        defer plus.deinit();
        plus.data[k] += h;
        try rsf.forwardCPU(&plus);
        var minus = try tensorClone(allocator, &input);
        defer minus.deinit();
        minus.data[k] -= h;
        try rsf.forwardCPU(&minus);
        var loss_plus: f32 = 0.0;
        var loss_minus: f32 = 0.0;
        var j: usize = 0;
        while (j < dim2) : (j += 1) {
            loss_plus += seeds.data[j] * plus.data[j];
            loss_minus += seeds.data[j] * minus.data[j];
        }
        const numeric = (loss_plus - loss_minus) / (2.0 * h);
        try std.testing.expect(valuesWithinTolerance(numeric, grad_input.data[k], 2.0e-3, 2.0e-2));
    }
}
test "RSF backward weight gradients match central finite differences" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    const dim2: usize = dim * 2;
    var rsf = try RSF.init(allocator, dim, 1);
    defer rsf.deinit();
    const core = rsf.ctrl orelse return error.MissingControl;
    var input = try Tensor.randomUniform(allocator, &[_]usize{ 1, dim2 }, -0.4, 0.4, 8123);
    defer input.deinit();
    var seeds = try Tensor.randomUniform(allocator, &[_]usize{ 1, dim2 }, -0.7, 0.7, 8124);
    defer seeds.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forwardCPU(&output);
    var grad_input = try Tensor.init(allocator, &[_]usize{ 1, dim2 });
    defer grad_input.deinit();
    try rsf.zeroGradients();
    try rsf.backward(&seeds, &input, &output, &grad_input);
    const s_grad = core.layers[0].s_weight_grad orelse return error.NoGradients;
    const t_grad = core.layers[0].t_weight_grad orelse return error.NoGradients;
    const h: f32 = 1.0e-3;
    var w: usize = 0;
    while (w < s_grad.data.len) : (w += 1) {
        const numeric_s = try lossSensitivity(&core.layers[0].s_weight, w, h, &rsf, &input, &seeds, allocator);
        try std.testing.expect(valuesWithinTolerance(numeric_s, s_grad.data[w], 2.0e-3, 2.0e-2));
        const numeric_t = try lossSensitivity(&core.layers[0].t_weight, w, h, &rsf, &input, &seeds, allocator);
        try std.testing.expect(valuesWithinTolerance(numeric_t, t_grad.data[w], 2.0e-3, 2.0e-2));
    }
}
fn lossSensitivity(weight: *Tensor, index: usize, h: f32, model: *RSF, input: *const Tensor, seeds: *const Tensor, allocator: Allocator) !f32 {
    const original = weight.data[index];
    defer weight.data[index] = original;
    weight.data[index] = original + h;
    var plus = try tensorClone(allocator, input);
    defer plus.deinit();
    try model.forwardCPU(&plus);
    weight.data[index] = original - h;
    var minus = try tensorClone(allocator, input);
    defer minus.deinit();
    try model.forwardCPU(&minus);
    var loss_plus: f32 = 0.0;
    var loss_minus: f32 = 0.0;
    var j: usize = 0;
    while (j < seeds.data.len) : (j += 1) {
        loss_plus += seeds.data[j] * plus.data[j];
        loss_minus += seeds.data[j] * minus.data[j];
    }
    return (loss_plus - loss_minus) / (2.0 * h);
}
test "RSF global registries release their storage once every core is destroyed" {
    const allocator = std.testing.allocator;
    var rsf = try RSF.init(allocator, 4, 1);
    try std.testing.expectError(error.ResourcesStillActive, shutdownGlobalRegistries());
    rsf.deinit();
    try shutdownGlobalRegistries();
    var reopened = try RSF.init(allocator, 4, 1);
    defer reopened.deinit();
    try std.testing.expect(reopened.id != 0);
}
test "RSF backward rejects an output that does not match the replayed forward pass" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    const dim2: usize = dim * 2;
    var rsf = try RSF.init(allocator, dim, 2);
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ 2, dim2 }, -0.3, 0.3, 6120);
    defer input.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forwardCPU(&output);
    output.data[0] += 5.0;
    var grad_output = try Tensor.randomUniform(allocator, &[_]usize{ 2, dim2 }, -0.2, 0.2, 6121);
    defer grad_output.deinit();
    var grad_input = try Tensor.init(allocator, &[_]usize{ 2, dim2 });
    defer grad_input.deinit();
    try std.testing.expectError(error.StateMismatch, rsf.backward(&grad_output, &input, &output, &grad_input));
}

test "RSF latent state carries bindings, roundtrips and accumulates log-det" {
    const allocator = std.testing.allocator;
    const model_dim: usize = 8;
    const batch: usize = 3;
    var rsf = try RSF.initWithConfig(allocator, model_dim, 2, .{ .global_diffusion = false });
    defer rsf.deinit();
    var x1 = try Tensor.randomUniform(allocator, &[_]usize{ batch, model_dim }, -0.4, 0.4, 4242);
    defer x1.deinit();
    var x2 = try Tensor.randomUniform(allocator, &[_]usize{ batch, model_dim }, -0.4, 0.4, 4243);
    defer x2.deinit();
    var state = try RSFLatentState.fromHalves(allocator, &rsf, &x1, &x2);
    defer state.deinit();

    try std.testing.expectEqual(@as(usize, 3), state.data.shape.dims.len);
    try std.testing.expectEqual(batch, state.data.shape.dims[0]);
    try std.testing.expectEqual(model_dim, state.data.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 2), state.data.shape.dims[2]);
    try std.testing.expectEqual(@as(f32, 0.0), state.log_det);
    try std.testing.expectEqual(@as(usize, batch * model_dim * 2), try state.blockedLength());
    try state.binding.requireSpace(.latent_state);
    try state.binding.requireModel(rsf.id);
    try state.binding.requireDim(model_dim);

    const even = try state.evenView();
    const odd = try state.oddView();
    for (0..batch) |b| {
        for (0..model_dim) |d| {
            try std.testing.expectEqual(x1.data[b * model_dim + d], try even.get(b, d));
            try std.testing.expectEqual(x2.data[b * model_dim + d], try odd.get(b, d));
        }
    }
    try std.testing.expectError(error.OutOfBounds, even.get(batch, 0));
    try std.testing.expectError(error.OutOfBounds, odd.get(0, model_dim));
    try std.testing.expectError(error.NonFinite, even.set(0, 0, std.math.inf(f32)));
    const row_buf = try allocator.alloc(f32, model_dim);
    defer allocator.free(row_buf);
    try even.copyRowTo(1, row_buf);
    for (0..model_dim) |d| try std.testing.expectEqual(x1.data[1 * model_dim + d], row_buf[d]);
    try std.testing.expectError(error.DataLengthMismatch, even.copyRowTo(0, row_buf[0..2]));

    var other = try RSF.initWithConfig(allocator, model_dim, 1, .{ .global_diffusion = false });
    defer other.deinit();
    try std.testing.expectError(types.RSFBindingError.RSFModelMismatch, state.requireModel(&other));
    const forward_logdet = try rsf.forwardLatentWithLogDet(&state);
    try std.testing.expect(std.math.isFinite(forward_logdet));
    const inverse_logdet = try rsf.inverseLatentWithLogDet(&state);
    try std.testing.expectApproxEqAbs(forward_logdet, inverse_logdet, 1e-4);
    for (0..batch) |b| {
        for (0..model_dim) |d| {
            try std.testing.expectApproxEqAbs(x1.data[b * model_dim + d], try even.get(b, d), 1e-4);
            try std.testing.expectApproxEqAbs(x2.data[b * model_dim + d], try odd.get(b, d), 1e-4);
        }
    }

    const relative = try state.roundtripError(&rsf, allocator);
    try std.testing.expect(relative < 1e-4);

    try state.forwardThrough(&rsf);
    try std.testing.expectApproxEqAbs(forward_logdet, state.log_det, 1e-5);
    try state.inverseThrough(&rsf);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), state.log_det, 1e-4);
    for (0..batch) |b| {
        for (0..model_dim) |d| {
            try std.testing.expectApproxEqAbs(x1.data[b * model_dim + d], try even.get(b, d), 1e-4);
        }
    }

    var zero_state = try RSFLatentState.init(allocator, &rsf, batch);
    defer zero_state.deinit();
    const zero_even = try zero_state.evenView();
    for (0..batch) |b| {
        for (0..model_dim) |d| try std.testing.expectEqual(@as(f32, 0.0), try zero_even.get(b, d));
    }
    try zero_even.set(0, 0, 0.25);
    try std.testing.expectEqual(@as(f32, 0.25), try zero_even.get(0, 0));
    var cloned = try zero_state.clone(allocator);
    defer cloned.deinit();
    try std.testing.expectEqual(@as(f32, 0.25), try (try cloned.evenView()).get(0, 0));
    var cloned_view = try cloned.evenView();
    try cloned_view.set(0, 0, 0.5);
    try std.testing.expectEqual(@as(f32, 0.5), try cloned_view.get(0, 0));
    try std.testing.expectEqual(@as(f32, 0.25), try zero_even.get(0, 0));
    try std.testing.expectError(error.InvalidBatchSize, RSFLatentState.init(allocator, &rsf, 0));
}

test "RSF optimizer surface reads and writes layer weights and gradients" {
    const allocator = std.testing.allocator;
    const model_dim: usize = 6;
    const num_layers: usize = 2;
    const per_layer = model_dim * coupling_width;
    var rsf = try RSF.initWithConfig(allocator, model_dim, num_layers, .{ .global_diffusion = false });
    defer rsf.deinit();
    try std.testing.expectEqual(model_dim, try rsf.dim());
    try std.testing.expectEqual(num_layers, try rsf.layerCount());
    try std.testing.expectEqual(false, try rsf.globalDiffusionEnabled());
    try std.testing.expectError(error.DiffusionDisabled, rsf.diffusionLayout());

    const s_buf = try allocator.alloc(f32, per_layer);
    defer allocator.free(s_buf);
    const t_buf = try allocator.alloc(f32, per_layer);
    defer allocator.free(t_buf);
    try rsf.readLayerWeights(0, s_buf, t_buf);
    try std.testing.expect(try exactSpectralNormRank2(s_buf, model_dim) <= LAYER_TARGET_SPECTRAL_NORM * (1.0 + 1e-6));
    try std.testing.expect(try exactSpectralNormRank2(t_buf, model_dim) <= LAYER_TARGET_SPECTRAL_NORM * (1.0 + 1e-6));
    try std.testing.expectError(error.LayerIndexOutOfBounds, rsf.readLayerWeights(num_layers, s_buf, t_buf));
    try std.testing.expectError(error.DataLengthMismatch, rsf.readLayerWeights(0, s_buf[0..2], t_buf));

    const replacement = try allocator.alloc(f32, per_layer);
    defer allocator.free(replacement);
    for (replacement, 0..) |*v, k| v.* = if (k % coupling_width == WEIGHT_COLUMN) @as(f32, 0.05) else @as(f32, 0.01);
    try rsf.writeLayerWeights(1, replacement, replacement);
    try rsf.readLayerWeights(1, s_buf, t_buf);
    for (0..per_layer) |k| {
        try std.testing.expectEqual(replacement[k], s_buf[k]);
        try std.testing.expectEqual(replacement[k], t_buf[k]);
    }
    const bad = try allocator.alloc(f32, per_layer);
    defer allocator.free(bad);
    @memcpy(bad, replacement);
    bad[0] = std.math.inf(f32);
    try std.testing.expectError(error.NonFinite, rsf.writeLayerWeights(0, bad, replacement));
    try std.testing.expectError(error.DataLengthMismatch, rsf.writeLayerWeights(0, replacement[0..2], replacement));

    try std.testing.expectError(error.NoGradients, rsf.readLayerGradients(0, s_buf, t_buf));
    const products = try allocator.alloc(f32, model_dim * 3);
    defer allocator.free(products);
    try std.testing.expectError(error.NoGradients, rsf.readLayerGradientProducts(0, .t, products));
    try rsf.ensureGradients();
    try rsf.zeroGradients();
    try rsf.readLayerGradients(0, s_buf, t_buf);
    for (0..per_layer) |k| try std.testing.expectEqual(@as(f32, 0.0), s_buf[k]);

    const block = try allocator.alloc(f32, model_dim * 3);
    defer allocator.free(block);
    try rsf.readLayerGaussNewton(0, block);
    for (block) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    try std.testing.expectError(error.InvalidDataLength, rsf.readLayerGaussNewton(0, block[0..2]));
    try std.testing.expectError(error.InvalidDataLength, rsf.readLayerGradientProducts(0, .s, products[0..2]));

    const s_add = try allocator.alloc(f32, per_layer);
    defer allocator.free(s_add);
    const t_add = try allocator.alloc(f32, per_layer);
    defer allocator.free(t_add);
    for (s_add, 0..) |*v, k| v.* = @as(f32, @floatFromInt(k + 1)) * 0.25;
    for (t_add, 0..) |*v, k| v.* = -@as(f32, @floatFromInt(k + 1)) * 0.125;
    try rsf.accumulateLayerGradients(0, s_add, t_add);
    try rsf.accumulateLayerGradients(0, s_add, t_add);
    try rsf.readLayerGradients(0, s_buf, t_buf);
    for (0..per_layer) |k| {
        try std.testing.expectApproxEqAbs(2.0 * s_add[k], s_buf[k], 1e-6);
        try std.testing.expectApproxEqAbs(2.0 * t_add[k], t_buf[k], 1e-6);
    }
    try std.testing.expectError(error.DataLengthMismatch, rsf.accumulateLayerGradients(0, s_add[0..2], t_add));
    const nan_add = try allocator.alloc(f32, per_layer);
    defer allocator.free(nan_add);
    @memcpy(nan_add, s_add);
    nan_add[1] = std.math.nan(f32);
    try std.testing.expectError(error.NonFinite, rsf.accumulateLayerGradients(0, nan_add, t_add));

    try rsf.readLayerGradientProducts(0, .s, products);
    for (0..model_dim) |d| {
        const g_w = 2.0 * s_add[d * coupling_width + WEIGHT_COLUMN];
        const g_b = 2.0 * s_add[d * coupling_width + BIAS_COLUMN];
        try std.testing.expectApproxEqAbs(g_w * g_w, products[d * 3 + 0], 1e-5);
        try std.testing.expectApproxEqAbs(g_w * g_b, products[d * 3 + 1], 1e-5);
        try std.testing.expectApproxEqAbs(g_b * g_b, products[d * 3 + 2], 1e-5);
    }
    try rsf.readLayerGradientProducts(0, .t, products);
    for (0..model_dim) |d| {
        const g_w = 2.0 * t_add[d * coupling_width + WEIGHT_COLUMN];
        const g_b = 2.0 * t_add[d * coupling_width + BIAS_COLUMN];
        try std.testing.expectApproxEqAbs(g_w * g_b, products[d * 3 + 1], 1e-5);
    }

    try rsf.scaleLayerGradients(0.5);
    try rsf.readLayerGradients(0, s_buf, t_buf);
    for (0..per_layer) |k| {
        try std.testing.expectApproxEqAbs(s_add[k], s_buf[k], 1e-6);
        try std.testing.expectApproxEqAbs(t_add[k], t_buf[k], 1e-6);
    }
    try std.testing.expectError(error.NonFinite, rsf.scaleLayerGradients(std.math.nan(f32)));
    try rsf.scaleLayerGradients(0.0);
    try rsf.readLayerGradients(0, s_buf, t_buf);
    for (0..per_layer) |k| try std.testing.expectEqual(@as(f32, 0.0), s_buf[k]);

    const bindings = try rsf.layerBindingsFor(0);
    try bindings.s_weight.requireSpace(.layer_weight_s);
    try bindings.s_weight.requireModel(rsf.id);
    try bindings.s_weight.requireLayer(0);
    try bindings.s_weight.requireDim(model_dim);
    try bindings.t_weight.requireSpace(.layer_weight_t);
    try bindings.gradient.requireSpace(.gradient);
    try bindings.fisher_block.requireSpace(.fisher_block);
    const latent_binding = try rsf.latentBinding();
    try latent_binding.requireSpace(.latent_state);
    try latent_binding.requireModel(rsf.id);
    try latent_binding.requireDim(model_dim);
    try std.testing.expectError(error.LayerIndexOutOfBounds, rsf.layerBindingsFor(num_layers));
}

test "RSF records gauss-newton blocks for the log-det objective" {
    const allocator = std.testing.allocator;
    const model_dim: usize = 4;
    const batch: usize = 1;
    var rsf = try RSF.initWithConfig(allocator, model_dim, 1, .{ .global_diffusion = false });
    defer rsf.deinit();
    var input = try Tensor.randomUniform(allocator, &[_]usize{ batch, 2 * model_dim }, -0.3, 0.3, 5150);
    defer input.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    var grad_output = try Tensor.randomUniform(allocator, &[_]usize{ batch, 2 * model_dim }, -0.2, 0.2, 5151);
    defer grad_output.deinit();
    var grad_input = try Tensor.init(allocator, &[_]usize{ batch, 2 * model_dim });
    defer grad_input.deinit();
    try rsf.zeroGradients();
    try rsf.backwardWithLogDet(&grad_output, &input, &output, &grad_input, 1.0);
    const gn_len = model_dim * 3;
    const block = try allocator.alloc(f32, gn_len);
    defer allocator.free(block);
    try rsf.readLayerGaussNewton(0, block);
    var positive: usize = 0;
    for (block) |v| {
        try std.testing.expect(std.math.isFinite(v));
        try std.testing.expect(v >= 0.0);
        if (v > 0.0) positive += 1;
    }
    try std.testing.expect(positive > 0);
    for (0..model_dim) |d| {
        try std.testing.expectApproxEqAbs(block[d * 3 + 0] * block[d * 3 + 2], block[d * 3 + 1] * block[d * 3 + 1], 1e-8);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), block[d * 3 + 2], 1e-6);
    }
    try rsf.zeroLayerGaussNewton(0);
    try rsf.readLayerGaussNewton(0, block);
    for (block) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    try rsf.zeroGradients();
    try rsf.backwardWithLogDet(&grad_output, &input, &output, &grad_input, 0.0);
    try rsf.readLayerGaussNewton(0, block);
    for (block) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    try rsf.zeroGradients();
    try rsf.backwardWithLogDet(&grad_output, &input, &output, &grad_input, 2.0);
    try rsf.readLayerGaussNewton(0, block);
    for (0..model_dim) |d| try std.testing.expectApproxEqAbs(@as(f32, 4.0), block[d * 3 + 2], 1e-5);
}

test "RSF save format v7 roundtrips the diffusion flag and gauss-newton blocks" {
    const allocator = std.testing.allocator;
    const model_dim: usize = 8;
    const num_layers: usize = 2;
    const batch: usize = 1;
    const per_layer = model_dim * coupling_width;
    const gn_len = model_dim * 3;
    var rsf = try RSF.initWithConfig(allocator, model_dim, num_layers, .{});
    defer rsf.deinit();
    try std.testing.expectEqual(true, try rsf.globalDiffusionEnabled());
    const layout = try rsf.diffusionLayout();
    try std.testing.expectEqual(2 * model_dim, layout.row_len);
    try std.testing.expect(layout.radix * layout.block == layout.row_len);

    var input = try Tensor.randomUniform(allocator, &[_]usize{ batch, 2 * model_dim }, -0.3, 0.3, 7070);
    defer input.deinit();
    var output = try tensorClone(allocator, &input);
    defer output.deinit();
    try rsf.forward(&output);
    var grad_output = try Tensor.randomUniform(allocator, &[_]usize{ batch, 2 * model_dim }, -0.2, 0.2, 7071);
    defer grad_output.deinit();
    var grad_input = try Tensor.init(allocator, &[_]usize{ batch, 2 * model_dim });
    defer grad_input.deinit();
    try rsf.zeroGradients();
    try rsf.backwardWithLogDet(&grad_output, &input, &output, &grad_input, 1.0);

    const block = try allocator.alloc(f32, gn_len);
    defer allocator.free(block);
    try rsf.readLayerGaussNewton(1, block);
    const s_buf = try allocator.alloc(f32, per_layer);
    defer allocator.free(s_buf);
    const t_buf = try allocator.alloc(f32, per_layer);
    defer allocator.free(t_buf);
    try rsf.readLayerWeights(1, s_buf, t_buf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_path);
    const file_path = try std.fs.path.join(allocator, &.{ dir_path, "model_v7.rsf" });
    defer allocator.free(file_path);
    try rsf.save(file_path);

    var loaded = try RSF.load(allocator, file_path);
    defer loaded.deinit();
    try std.testing.expectEqual(model_dim, try loaded.dim());
    try std.testing.expectEqual(num_layers, try loaded.layerCount());
    try std.testing.expectEqual(true, try loaded.globalDiffusionEnabled());
    const loaded_layout = try loaded.diffusionLayout();
    try std.testing.expectEqual(layout.row_len, loaded_layout.row_len);
    try std.testing.expectEqual(layout.radix, loaded_layout.radix);
    try std.testing.expectEqual(layout.block, loaded_layout.block);
    try std.testing.expectEqual(layout.stages, loaded_layout.stages);

    const loaded_block = try allocator.alloc(f32, gn_len);
    defer allocator.free(loaded_block);
    try loaded.readLayerGaussNewton(1, loaded_block);
    for (0..gn_len) |k| try std.testing.expectEqual(block[k], loaded_block[k]);
    const loaded_s = try allocator.alloc(f32, per_layer);
    defer allocator.free(loaded_s);
    const loaded_t = try allocator.alloc(f32, per_layer);
    defer allocator.free(loaded_t);
    try loaded.readLayerWeights(1, loaded_s, loaded_t);
    for (0..per_layer) |k| {
        try std.testing.expectEqual(s_buf[k], loaded_s[k]);
        try std.testing.expectEqual(t_buf[k], loaded_t[k]);
    }
    try std.testing.expect(try loaded.verifyInvertible(&input, 1e-4, 1e-4));
    try std.testing.expectError(error.PolicyMismatch, RSF.loadWithConfig(allocator, file_path, .{ .global_diffusion = false }));
    var policy_loaded = try RSF.loadWithConfig(allocator, file_path, .{ .global_diffusion = true });
    defer policy_loaded.deinit();
    try std.testing.expectEqual(true, try policy_loaded.globalDiffusionEnabled());
    try std.testing.expect(try RSF.saveLoadRoundtrip(allocator, &rsf, file_path, 0.0, 0.0));
}

test "RSF loads a v6 snapshot without diffusion and with zero gauss-newton blocks" {
    const allocator = std.testing.allocator;
    const model_dim: usize = 4;
    const num_layers: usize = 2;
    const per_layer = model_dim * coupling_width;
    const cfg = RSFConfig{
        .clip_min = -5.0,
        .clip_max = 5.0,
        .grad_mean = true,
        .max_dim = 1 << 20,
        .max_layers = 1 << 20,
        .global_diffusion = false,
    };
    var source = try RSF.initWithConfig(allocator, model_dim, num_layers, cfg);
    defer source.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(dir_path);
    const file_path = try std.fs.path.join(allocator, &.{ dir_path, "legacy_v6.rsf" });
    defer allocator.free(file_path);

    const s_buf = try allocator.alloc(f32, per_layer);
    defer allocator.free(s_buf);
    const t_buf = try allocator.alloc(f32, per_layer);
    defer allocator.free(t_buf);

    const file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    var buffered = std.io.bufferedWriter(file.writer());
    const w = buffered.writer();
    var hasher = std.hash.Crc32.init();
    try w.writeAll("RSF0");
    hasher.update("RSF0");
    try w.writeInt(u32, SAVE_VERSION_LEGACY, .little);
    crcUpdateU32LE(&hasher, SAVE_VERSION_LEGACY);
    try w.writeInt(u64, @intCast(num_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(num_layers));
    try w.writeInt(u64, @intCast(model_dim), .little);
    crcUpdateU64LE(&hasher, @intCast(model_dim));
    const clip_min_bits = @as(u32, @bitCast(cfg.clip_min));
    const clip_max_bits = @as(u32, @bitCast(cfg.clip_max));
    try w.writeInt(u32, clip_min_bits, .little);
    try w.writeInt(u32, clip_max_bits, .little);
    crcUpdateU32LE(&hasher, clip_min_bits);
    crcUpdateU32LE(&hasher, clip_max_bits);
    const gm_byte: u8 = if (cfg.grad_mean) 1 else 0;
    try w.writeByte(gm_byte);
    crcUpdateU8(&hasher, gm_byte);
    try w.writeInt(u64, @intCast(cfg.max_dim), .little);
    try w.writeInt(u64, @intCast(cfg.max_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(cfg.max_dim));
    crcUpdateU64LE(&hasher, @intCast(cfg.max_layers));
    var l: usize = 0;
    while (l < num_layers) : (l += 1) {
        try source.readLayerWeights(l, s_buf, t_buf);
        try w.writeInt(u32, clip_min_bits, .little);
        try w.writeInt(u32, clip_max_bits, .little);
        crcUpdateU32LE(&hasher, clip_min_bits);
        crcUpdateU32LE(&hasher, clip_max_bits);
        try w.writeByte(gm_byte);
        crcUpdateU8(&hasher, gm_byte);
        var s_w = try Tensor.init(allocator, &[_]usize{ model_dim, coupling_width });
        defer s_w.deinit();
        @memcpy(s_w.data, s_buf);
        var t_w = try Tensor.init(allocator, &[_]usize{ model_dim, coupling_width });
        defer t_w.deinit();
        @memcpy(t_w.data, t_buf);
        try writeTensorData(w, &hasher, &s_w);
        try writeTensorData(w, &hasher, &t_w);
    }
    try w.writeInt(u32, hasher.final(), .little);
    try buffered.flush();
    try file.sync();

    var loaded = try RSF.load(allocator, file_path);
    defer loaded.deinit();
    try std.testing.expectEqual(model_dim, try loaded.dim());
    try std.testing.expectEqual(num_layers, try loaded.layerCount());
    try std.testing.expectEqual(false, try loaded.globalDiffusionEnabled());
    try std.testing.expectError(error.DiffusionDisabled, loaded.diffusionLayout());
    const gn = try allocator.alloc(f32, model_dim * 3);
    defer allocator.free(gn);
    const s_out = try allocator.alloc(f32, per_layer);
    defer allocator.free(s_out);
    const t_out = try allocator.alloc(f32, per_layer);
    defer allocator.free(t_out);
    for (0..num_layers) |i| {
        try loaded.readLayerGaussNewton(i, gn);
        for (gn) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
        try loaded.readLayerWeights(i, s_out, t_out);
        try source.readLayerWeights(i, s_buf, t_buf);
        for (0..per_layer) |k| {
            try std.testing.expectEqual(s_buf[k], s_out[k]);
            try std.testing.expectEqual(t_buf[k], t_out[k]);
        }
    }
    try std.testing.expectError(error.PolicyMismatch, RSF.loadWithConfig(allocator, file_path, .{ .global_diffusion = true }));
    var policy_loaded = try RSF.loadWithConfig(allocator, file_path, cfg);
    defer policy_loaded.deinit();
    try std.testing.expectEqual(false, try policy_loaded.globalDiffusionEnabled());
}
