const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../core/tensor.zig");
const types = @import("../core/types.zig");
const Tensor = tensor.Tensor;

pub const OFTB = struct {
    pub const FRACTAL_SCALE: f32 = 0.7071067811865476;
    pub const FRACTAL_SCALE_SQ: f32 = 0.5000000000000001;
    pub const LOG_DET_JACOBIAN: f32 = 0.0;
    pub const resonance_order: usize = 8;

    dim: usize,
    global_diffusion: bool,
    layout: ?types.RSFDiffusionLayout,

    pub fn init(d: usize) OFTB {
        std.debug.assert(d != 0);
        std.debug.assert(d <= std.math.maxInt(usize) / 2);
        return OFTB{
            .dim = d,
            .global_diffusion = false,
            .layout = null,
        };
    }

    pub fn initWithDiffusion(d: usize, global_diffusion: bool) OFTB {
        std.debug.assert(d != 0);
        std.debug.assert(d <= std.math.maxInt(usize) / 2);
        if (!global_diffusion) {
            return OFTB{
                .dim = d,
                .global_diffusion = false,
                .layout = null,
            };
        }
        const row_len = d * 2;
        const maybe_layout = types.rsfDiffusionLayout(row_len);
        if (maybe_layout == null) {
            return OFTB{
                .dim = d,
                .global_diffusion = false,
                .layout = null,
            };
        }
        const layout = maybe_layout.?;
        std.debug.assert(tensor.diffusionLayoutIsApplicable(row_len, layout));
        return OFTB{
            .dim = d,
            .global_diffusion = true,
            .layout = layout,
        };
    }

    pub fn deinit(self: *OFTB) void {
        self.* = undefined;
    }

    pub fn diffusionEnabled(self: OFTB) bool {
        return self.global_diffusion and self.layout != null;
    }

    pub fn diffusionLayout(self: OFTB) ?types.RSFDiffusionLayout {
        return self.layout;
    }

    pub fn rowLen(self: OFTB) usize {
        return self.dim * 2;
    }

    fn vectorLen() usize {
        if (comptime builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
            return 16;
        }
        return 8;
    }

    pub fn applyRotationSliceInPlace(self: OFTB, data: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (data.len != total) return;
        const half = self.dim;
        const x1 = data[0..half];
        const x2 = data[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = x1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = x2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            x1[i..][0..VLEN].* = (va - vb) * vscale;
            x2[i..][0..VLEN].* = (va + vb) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = x1[i];
            const b = x2[i];
            x1[i] = (a - b) * scale;
            x2[i] = (a + b) * scale;
        }
    }

    pub fn applyRotationAdjointSliceInPlace(self: OFTB, data: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (data.len != total) return;
        const half = self.dim;
        const g1 = data[0..half];
        const g2 = data[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = g1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = g2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            g1[i..][0..VLEN].* = (va + vb) * vscale;
            g2[i..][0..VLEN].* = (vb - va) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = g1[i];
            const b = g2[i];
            g1[i] = (a + b) * scale;
            g2[i] = (b - a) * scale;
        }
    }

    pub fn applyDiffusionSliceInPlace(self: OFTB, data: []f32) void {
        const layout = self.layout orelse return;
        if (data.len != self.dim * 2) return;
        tensor.globalDiffuseRowUnchecked(data, layout);
    }

    pub fn applyDiffusionRowsInPlace(self: OFTB, rows: []f32, count: usize) void {
        const layout = self.layout orelse return;
        const total = self.dim * 2;
        var r: usize = 0;
        while (r < count) : (r += 1) {
            const base = r * total;
            if (base + total > rows.len) return;
            tensor.globalDiffuseRowUnchecked(rows[base .. base + total], layout);
        }
    }

    pub fn forwardInPlace(self: OFTB, x: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (x.data.len != total) return error.DimensionMismatch;
        self.forwardSliceInPlace(x.data);
    }

    pub fn forwardSliceInPlace(self: OFTB, data: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (data.len != total) return;
        self.applyRotationSliceInPlace(data);
        self.applyDiffusionSliceInPlace(data);
    }

    pub fn backwardInPlace(self: OFTB, grad: []f32) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (grad.len != total) return error.DimensionMismatch;
        self.backwardSliceInPlace(grad);
    }

    pub fn backwardSliceInPlace(self: OFTB, grad: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (grad.len != total) return;
        self.applyDiffusionSliceInPlace(grad);
        self.applyRotationAdjointSliceInPlace(grad);
    }

    pub fn inverseInPlace(self: OFTB, x: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (x.data.len != total) return error.DimensionMismatch;
        self.backwardSliceInPlace(x.data);
    }

    pub fn inverseSliceInPlace(self: OFTB, data: []f32) void {
        self.backwardSliceInPlace(data);
    }

    pub fn forwardBackwardFusedInPlace(self: OFTB, activation: []f32, grad: []f32) void {
        self.forwardSliceInPlace(activation);
        self.backwardSliceInPlace(grad);
    }

    pub fn symplecticReversalInPlace(self: OFTB, activation: []f32, grad: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (activation.len != total or grad.len != total) return;
        self.applyDiffusionSliceInPlace(activation);
        self.applyDiffusionSliceInPlace(grad);
        const half = self.dim;
        const a1 = activation[0..half];
        const a2 = activation[half..][0..half];
        const g1 = grad[0..half];
        const g2 = grad[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const wa: @Vector(VLEN, f32) = a1[i..][0..VLEN].*;
            const wb: @Vector(VLEN, f32) = a2[i..][0..VLEN].*;
            const wscale: @Vector(VLEN, f32) = @splat(scale);
            a1[i..][0..VLEN].* = (wa + wb) * wscale;
            a2[i..][0..VLEN].* = (wb - wa) * wscale;
            const va: @Vector(VLEN, f32) = g1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = g2[i..][0..VLEN].*;
            g1[i..][0..VLEN].* = (va + vb) * wscale;
            g2[i..][0..VLEN].* = (vb - va) * wscale;
        }
        while (i < half) : (i += 1) {
            const a = a1[i];
            const b = a2[i];
            a1[i] = (a + b) * scale;
            a2[i] = (b - a) * scale;
            const ga = g1[i];
            const gb = g2[i];
            g1[i] = (ga + gb) * scale;
            g2[i] = (gb - ga) * scale;
        }
    }

    pub fn logDeterminantJacobian(self: OFTB) f32 {
        _ = self;
        return LOG_DET_JACOBIAN;
    }

    pub fn logDeterminantAdjointShift(_: OFTB) f32 {
        return 1.0;
    }

    pub fn isSymplectic(_: OFTB) bool {
        return true;
    }

    pub fn isOrthogonal(_: OFTB) bool {
        return true;
    }

    pub fn diffusionIsInvolution(self: OFTB) bool {
        return self.diffusionEnabled();
    }

    pub fn resonanceOrder(_: OFTB) usize {
        return resonance_order;
    }

    pub fn forwardRows(self: OFTB, rows: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (rows.shape.dims.len != 2) return error.DimensionMismatch;
        const total = self.dim * 2;
        if (rows.shape.dims[1] != total) return error.DimensionMismatch;
        if (!rows.shape.isContiguous()) return error.DimensionMismatch;
        const row_count = rows.shape.dims[0];
        var r: usize = 0;
        while (r < row_count) : (r += 1) {
            self.forwardSliceInPlace(rows.data[r * total ..][0..total]);
        }
    }

    pub fn inverseRows(self: OFTB, rows: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (rows.shape.dims.len != 2) return error.DimensionMismatch;
        const total = self.dim * 2;
        if (rows.shape.dims[1] != total) return error.DimensionMismatch;
        if (!rows.shape.isContiguous()) return error.DimensionMismatch;
        const row_count = rows.shape.dims[0];
        var r: usize = 0;
        while (r < row_count) : (r += 1) {
            self.backwardSliceInPlace(rows.data[r * total ..][0..total]);
        }
    }
};

pub fn mixForward(oftb: OFTB, x: *Tensor) !void {
    try oftb.forwardInPlace(x);
}

pub fn mixBackward(oftb: OFTB, grad: []f32) !void {
    try oftb.backwardInPlace(grad);
}

pub fn mixInverse(oftb: OFTB, x: *Tensor) !void {
    try oftb.inverseInPlace(x);
}

comptime {
    _ = OFTB;
}

fn oftbNormL2(values: []const f32) f64 {
    var acc: f64 = 0.0;
    for (values) |v| {
        const x: f64 = @floatCast(v);
        acc += x * x;
    }
    return @sqrt(acc);
}

fn oftbMaxAbsDiff(a: []const f32, b: []const f32) f32 {
    var worst: f32 = 0.0;
    for (a, b) |x, y| {
        const diff = @abs(x - y);
        if (diff > worst) worst = diff;
    }
    return worst;
}

test "OFTB forward then backward returns input within 1e-5 tolerance" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const dim: usize = 32;
    const total = dim * 2;

    var input = try Tensor.init(allocator, &.{ 1, total });
    defer input.deinit();
    var i: usize = 0;
    while (i < input.data.len) : (i += 1) {
        input.data[i] = (random.float(f32) - 0.5) * 2.0;
    }

    const original = try allocator.dupe(f32, input.data);
    defer allocator.free(original);

    var oftb = OFTB.init(dim);
    try oftb.forwardInPlace(&input);
    try oftb.backwardInPlace(input.data);

    i = 0;
    while (i < input.data.len) : (i += 1) {
        try std.testing.expectApproxEqAbs(input.data[i], original[i], 1e-5);
    }
}

test "OFTB inverse equals forward transpose and restores activations" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    const dim: usize = 64;
    const total = dim * 2;

    var input = try Tensor.init(allocator, &.{ 1, total });
    defer input.deinit();
    var i: usize = 0;
    while (i < input.data.len) : (i += 1) {
        input.data[i] = (random.float(f32) - 0.5) * 4.0;
    }

    const original = try allocator.dupe(f32, input.data);
    defer allocator.free(original);

    var oftb = OFTB.init(dim);
    try oftb.forwardInPlace(&input);
    try oftb.inverseInPlace(&input);

    i = 0;
    while (i < input.data.len) : (i += 1) {
        try std.testing.expectApproxEqAbs(input.data[i], original[i], 1e-5);
    }
    try std.testing.expectApproxEqAbs(oftb.logDeterminantJacobian(), 0.0, 1e-7);
}

test "OFTB symplectic reversal acts on activation and gradient in one pass" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const total = dim * 2;

    var act = try Tensor.init(allocator, &.{ 1, total });
    defer act.deinit();
    var grad_data = try allocator.alloc(f32, total);
    defer allocator.free(grad_data);

    var i: usize = 0;
    while (i < total) : (i += 1) {
        act.data[i] = @as(f32, @floatFromInt(i)) * 0.05 - 0.4;
        grad_data[i] = @as(f32, @floatFromInt(total - i)) * 0.02;
    }

    const act_copy = try allocator.dupe(f32, act.data);
    defer allocator.free(act_copy);
    const grad_copy = try allocator.dupe(f32, grad_data);
    defer allocator.free(grad_copy);

    var oftb = OFTB.init(dim);
    oftb.symplecticReversalInPlace(act.data, grad_data);
    oftb.forwardSliceInPlace(act.data);
    oftb.forwardSliceInPlace(grad_data);

    i = 0;
    while (i < total) : (i += 1) {
        try std.testing.expectApproxEqAbs(act.data[i], act_copy[i], 1e-5);
        try std.testing.expectApproxEqAbs(grad_data[i], grad_copy[i], 1e-5);
    }
}

test "OFTB legacy init disables diffusion and matches the scalar rotation" {
    const dim: usize = 12;
    const total = dim * 2;
    var data: [total]f32 = undefined;
    var reference: [total]f32 = undefined;
    var i: usize = 0;
    while (i < total) : (i += 1) {
        data[i] = @as(f32, @floatFromInt(i)) * 0.13 - 1.7;
        reference[i] = data[i];
    }
    const oftb = OFTB.init(dim);
    try std.testing.expect(!oftb.diffusionEnabled());
    try std.testing.expect(oftb.diffusionLayout() == null);
    oftb.forwardSliceInPlace(&data);
    const scale = OFTB.FRACTAL_SCALE;
    i = 0;
    while (i < dim) : (i += 1) {
        const a = reference[i];
        const b = reference[dim + i];
        reference[i] = (a - b) * scale;
        reference[dim + i] = (a + b) * scale;
    }
    try std.testing.expectEqualSlices(f32, &reference, &data);
    oftb.backwardSliceInPlace(&data);
    i = 0;
    while (i < total) : (i += 1) {
        try std.testing.expectApproxEqAbs(data[i], @as(f32, @floatFromInt(i)) * 0.13 - 1.7, 1e-5);
    }
}

test "OFTB rotation has order eight" {
    const dim: usize = 20;
    const total = dim * 2;
    var data: [total]f32 = undefined;
    var original: [total]f32 = undefined;
    var i: usize = 0;
    while (i < total) : (i += 1) {
        data[i] = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 2.0;
        original[i] = data[i];
    }
    const oftb = OFTB.init(dim);
    try std.testing.expectEqual(@as(usize, 8), oftb.resonanceOrder());
    var step: usize = 0;
    while (step < OFTB.resonance_order) : (step += 1) {
        oftb.forwardSliceInPlace(&data);
    }
    try std.testing.expect(oftbMaxAbsDiff(&original, &data) < 1e-4);
    try std.testing.expect(oftb.isOrthogonal());
    try std.testing.expect(oftb.isSymplectic());
    try std.testing.expectApproxEqAbs(oftb.logDeterminantJacobian(), 0.0, 1e-7);
    step = 0;
    while (step < 4) : (step += 1) oftb.forwardSliceInPlace(&data);
    try std.testing.expect(oftbMaxAbsDiff(&original, &data) > 1e-3);
}

test "OFTB diffusion operator is an exact involution" {
    const allocator = std.testing.allocator;
    const dim: usize = 24;
    const total = dim * 2;
    const oftb = OFTB.initWithDiffusion(dim, true);
    try std.testing.expect(oftb.diffusionEnabled());
    const layout = oftb.diffusionLayout().?;
    try std.testing.expectEqual(@as(usize, 3), layout.radix);
    try std.testing.expectEqual(@as(usize, 16), layout.block);
    try std.testing.expectEqual(@as(usize, 4), layout.stages);
    try std.testing.expectEqual(@as(usize, total), oftb.rowLen());

    const data = try allocator.alloc(f32, total);
    defer allocator.free(data);
    const original = try allocator.alloc(f32, total);
    defer allocator.free(original);
    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();
    for (data) |*v| {
        v.* = (random.float(f32) - 0.5) * 3.0;
    }
    @memcpy(original, data);
    const norm_before = oftbNormL2(data);
    oftb.applyDiffusionSliceInPlace(data);
    try std.testing.expectApproxEqRel(oftbNormL2(data), norm_before, 1e-5);
    oftb.applyDiffusionSliceInPlace(data);
    try std.testing.expect(oftbMaxAbsDiff(original, data) < 1e-4);
    try std.testing.expect(oftb.diffusionIsInvolution());

    const delta = try allocator.alloc(f32, total);
    defer allocator.free(delta);
    @memset(delta, 0.0);
    delta[5] = 1.0;
    oftb.applyDiffusionSliceInPlace(delta);
    var reached: usize = 0;
    for (0..layout.radix) |b| {
        var energy: f64 = 0.0;
        for (0..layout.block) |o| {
            const v: f64 = @floatCast(delta[b * layout.block + o]);
            energy += v * v;
        }
        if (energy > 1e-6) reached += 1;
    }
    try std.testing.expectEqual(layout.radix, reached);
}

test "OFTB layer map with diffusion stays orthogonal and invertible" {
    const allocator = std.testing.allocator;
    const dim: usize = 24;
    const total = dim * 2;
    const oftb = OFTB.initWithDiffusion(dim, true);
    const data = try allocator.alloc(f32, total);
    defer allocator.free(data);
    const original = try allocator.alloc(f32, total);
    defer allocator.free(original);
    var prng = std.Random.DefaultPrng.init(99);
    const random = prng.random();
    for (data) |*v| v.* = (random.float(f32) - 0.5) * 4.0;
    @memcpy(original, data);
    const norm_before = oftbNormL2(data);
    oftb.forwardSliceInPlace(data);
    try std.testing.expectApproxEqRel(oftbNormL2(data), norm_before, 1e-5);
    oftb.backwardSliceInPlace(data);
    try std.testing.expect(oftbMaxAbsDiff(original, data) < 1e-4);
    try std.testing.expectApproxEqAbs(oftb.logDeterminantJacobian(), 0.0, 1e-7);

    @memcpy(data, original);
    oftb.forwardSliceInPlace(data);
    const after_forward = try allocator.dupe(f32, data);
    defer allocator.free(after_forward);
    oftb.symplecticReversalInPlace(data, after_forward);
    try std.testing.expect(oftbMaxAbsDiff(original, data) < 1e-4);
    try std.testing.expect(oftbMaxAbsDiff(original, after_forward) < 1e-4);
}

test "OFTB diffusion removes cross channel isolation on the production geometry" {
    const allocator = std.testing.allocator;
    const dim: usize = 49152;
    const total = dim * 2;
    const oftb = OFTB.initWithDiffusion(dim, true);
    const layout = oftb.diffusionLayout().?;
    try std.testing.expectEqual(@as(usize, 98304), layout.row_len);
    try std.testing.expectEqual(@as(usize, 3), layout.radix);
    try std.testing.expectEqual(@as(usize, 32768), layout.block);
    try std.testing.expectEqual(@as(usize, 15), layout.stages);

    const data = try allocator.alloc(f32, total);
    defer allocator.free(data);
    const perturbed = try allocator.alloc(f32, total);
    defer allocator.free(perturbed);
    const original = try allocator.alloc(f32, total);
    defer allocator.free(original);
    var prng = std.Random.DefaultPrng.init(20264);
    const random = prng.random();
    for (data) |*v| v.* = (random.float(f32) - 0.5) * 2.0;
    @memcpy(perturbed, data);
    perturbed[1234] += 0.25;
    @memcpy(original, data);

    const norm_before = oftbNormL2(data);
    oftb.forwardSliceInPlace(data);
    oftb.forwardSliceInPlace(perturbed);
    try std.testing.expectApproxEqRel(oftbNormL2(data), norm_before, 1e-5);
    for (0..layout.radix) |b| {
        const base = b * layout.block;
        var block_delta: f64 = 0.0;
        for (0..layout.block) |o| {
            block_delta += @abs(@as(f64, @floatCast(data[base + o])) - @as(f64, @floatCast(perturbed[base + o])));
        }
        try std.testing.expect(block_delta > 1e-3);
    }
    var changed: usize = 0;
    for (data, perturbed) |x, y| {
        if (@abs(x - y) > 1e-9) changed += 1;
    }
    try std.testing.expect(changed >= total / 2);
    try std.testing.expect(changed < total);

    oftb.backwardSliceInPlace(data);
    try std.testing.expect(oftbMaxAbsDiff(original, data) < 1e-3);

    const delta = try allocator.alloc(f32, total);
    defer allocator.free(delta);
    @memset(delta, 0.0);
    delta[1234] = 1.0;
    oftb.applyDiffusionSliceInPlace(delta);
    try std.testing.expectApproxEqRel(oftbNormL2(delta), @as(f64, 1.0), 1e-5);
    for (0..layout.radix) |b| {
        var energy: f64 = 0.0;
        for (0..layout.block) |o| {
            const v: f64 = @floatCast(delta[b * layout.block + o]);
            energy += v * v;
        }
        try std.testing.expect(energy > 1e-6);
    }
    var nonzero: usize = 0;
    for (delta) |v| {
        if (@abs(v) > 1e-9) nonzero += 1;
    }
    try std.testing.expect(nonzero >= total / 2);
}

test "OFTB row batch APIs apply diffusion per row" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const total = dim * 2;
    const rows: usize = 4;
    const oftb = OFTB.initWithDiffusion(dim, true);
    var batch = try Tensor.init(allocator, &.{ rows, total });
    defer batch.deinit();
    var prng = std.Random.DefaultPrng.init(555);
    const random = prng.random();
    for (batch.data) |*v| v.* = (random.float(f32) - 0.5) * 2.0;
    const original = try allocator.dupe(f32, batch.data);
    defer allocator.free(original);

    const single = try allocator.alloc(f32, total);
    defer allocator.free(single);

    try oftb.forwardRows(&batch);
    for (0..rows) |r| {
        @memcpy(single, original[r * total ..][0..total]);
        oftb.forwardSliceInPlace(single);
        try std.testing.expectEqualSlices(f32, single, batch.data[r * total ..][0..total]);
    }
    try oftb.inverseRows(&batch);
    try std.testing.expect(oftbMaxAbsDiff(original, batch.data) < 1e-4);

    var wrong = try Tensor.init(allocator, &.{ rows, total + 2 });
    defer wrong.deinit();
    try std.testing.expectError(error.DimensionMismatch, oftb.forwardRows(&wrong));
    var flat = try Tensor.init(allocator, &.{rows * total});
    defer flat.deinit();
    try std.testing.expectError(error.DimensionMismatch, oftb.forwardRows(&flat));
}

test "OFTB initWithDiffusion false reproduces the legacy map" {
    const dim: usize = 18;
    const total = dim * 2;
    var a: [total]f32 = undefined;
    var b: [total]f32 = undefined;
    var i: usize = 0;
    while (i < total) : (i += 1) {
        a[i] = @cos(@as(f32, @floatFromInt(i)) * 0.21) * 1.5;
        b[i] = a[i];
    }
    const legacy = OFTB.init(dim);
    const disabled = OFTB.initWithDiffusion(dim, false);
    try std.testing.expect(!disabled.diffusionEnabled());
    legacy.forwardSliceInPlace(&a);
    disabled.forwardSliceInPlace(&b);
    try std.testing.expectEqualSlices(f32, &a, &b);
    legacy.backwardSliceInPlace(&a);
    disabled.backwardSliceInPlace(&b);
    try std.testing.expectEqualSlices(f32, &a, &b);
}

test "OFTB slice APIs ignore mismatched lengths" {
    const dim: usize = 8;
    const oftb = OFTB.initWithDiffusion(dim, true);
    var short: [8]f32 = [_]f32{1} ** 8;
    var copy: [8]f32 = [_]f32{1} ** 8;
    oftb.forwardSliceInPlace(&short);
    try std.testing.expectEqualSlices(f32, &copy, &short);
    oftb.backwardSliceInPlace(&short);
    try std.testing.expectEqualSlices(f32, &copy, &short);
    oftb.applyDiffusionSliceInPlace(&short);
    try std.testing.expectEqualSlices(f32, &copy, &short);
    const legacy = OFTB.init(dim);
    legacy.applyDiffusionSliceInPlace(&short);
    try std.testing.expectEqualSlices(f32, &copy, &short);
    const allocator = std.testing.allocator;
    var mismatched = try Tensor.init(allocator, &.{ 1, dim });
    defer mismatched.deinit();
    try std.testing.expectError(error.DimensionMismatch, legacy.forwardInPlace(&mismatched));
    try std.testing.expectError(error.DimensionMismatch, legacy.backwardInPlace(mismatched.data));
    try std.testing.expectError(error.DimensionMismatch, legacy.inverseInPlace(&mismatched));
}
