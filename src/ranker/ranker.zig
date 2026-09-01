const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const BitSet = types.BitSet;
const Tensor = @import("../core/tensor.zig").Tensor;
const SSI = @import("../index/ssi.zig").SSI;
const core_io = @import("../core/io.zig");
const stableHash = core_io.stableHash;
const Error = types.Error;

pub const RankerConfig = struct {
    pub const STREAMING_BUFFER_SIZE: usize = 1024;
    pub const STREAMING_WINDOW_SIZE: usize = 512;
    pub const DEFAULT_TOP_N_RETRIEVAL: usize = 1000;
    pub const HASH_SEED_MULTIPLIER_A: u64 = 0x9e3779b97f4a7c15;
    pub const HASH_SEED_MULTIPLIER_B: u64 = 0x517cc1b727220a95;
    pub const LEARNING_RATE: f32 = 0.01;
    pub const DIVERSITY_WEIGHT: f32 = 0.3;
    pub const PROXIMITY_WEIGHT: f32 = 0.3;
    pub const MAX_RAW_SCORE: f32 = 100.0;
    pub const BASE_SCORE_WEIGHT: f32 = 0.4;
    pub const OVERLAP_WEIGHT: f32 = 0.3;
    pub const JACCARD_WEIGHT: f32 = 0.3;
    pub const SCORE_RETRIEVAL_LIMIT: usize = 32;
    pub const MAX_NGRAM_ORDER: usize = 64;
    pub const PROXIMITY_SCALE: u64 = 1024;
    pub const SCORE_SIGMOID_CENTER: f32 = 0.8;
    pub const SCORE_SIGMOID_WIDTH: f32 = 0.4;
};

fn tokenToLEBytes(token: u32) [4]u8 {
    return mem.toBytes(mem.nativeToLittle(u32, token));
}

fn encodeNgramLE(ngram: []const u32, buf: []u8) void {
    for (ngram, 0..) |token, i| {
        const le = tokenToLEBytes(token);
        buf[i * 4 + 0] = le[0];
        buf[i * 4 + 1] = le[1];
        buf[i * 4 + 2] = le[2];
        buf[i * 4 + 3] = le[3];
    }
}

fn sigmoidScale(raw: f32) f32 {
    const z = (raw - RankerConfig.SCORE_SIGMOID_CENTER) / RankerConfig.SCORE_SIGMOID_WIDTH;
    if (z >= 0.0) {
        return 1.0 / (1.0 + @exp(-z));
    } else {
        const e = @exp(z);
        return e / (1.0 + e);
    }
}

fn sigmoidDerivative(out: f32) f32 {
    return out * (1.0 - out) / RankerConfig.SCORE_SIGMOID_WIDTH;
}

fn freeRankedSegments(segments: []types.RankedSegment, allocator: Allocator) void {
    for (segments) |*seg| seg.deinit(allocator);
    allocator.free(segments);
}

const RankedSegmentComparator = struct {
    pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
        if (math.isNan(a.score) and math.isNan(b.score)) return .eq;
        if (math.isNan(a.score)) return .lt;
        if (math.isNan(b.score)) return .gt;
        return std.math.order(a.score, b.score);
    }
};

pub const Ranker = struct {
    ngram_weights: []f32,
    lsh_hash_params: []u64,
    num_hash_functions: usize,
    num_ngrams: usize,
    seed: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_ngrams: usize, num_hash_funcs: usize, seed: u64) !Ranker {
        if (num_ngrams == 0) return error.InvalidParameter;
        if (num_ngrams > RankerConfig.MAX_NGRAM_ORDER) return error.InvalidParameter;
        if (num_hash_funcs == 0) return error.InvalidParameter;

        const weights = try allocator.alloc(f32, num_ngrams);
        errdefer allocator.free(weights);

        var i: usize = 0;
        while (i < weights.len) : (i += 1) {
            const decay = 1.0 / @as(f32, @floatFromInt(i + 1));
            weights[i] = decay;
        }

        const hash_params = try allocator.alloc(u64, num_hash_funcs * 2);
        errdefer allocator.free(hash_params);

        i = 0;
        while (i < num_hash_funcs) : (i += 1) {
            const i_u64: u64 = @intCast(i);
            const i_plus_one: u64 = @intCast(i + 1);
            hash_params[i * 2] = seed +% (i_u64 *% RankerConfig.HASH_SEED_MULTIPLIER_A);
            hash_params[i * 2 + 1] = seed +% (i_plus_one *% RankerConfig.HASH_SEED_MULTIPLIER_B);
        }

        return .{
            .ngram_weights = weights,
            .lsh_hash_params = hash_params,
            .num_hash_functions = num_hash_funcs,
            .num_ngrams = num_ngrams,
            .seed = seed,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Ranker) void {
        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
    }

    fn windowHash(self: *const Ranker, window: []const u32) u64 {
        const cap_tokens = @min(window.len, RankerConfig.MAX_NGRAM_ORDER);
        var buf: [RankerConfig.MAX_NGRAM_ORDER * 4]u8 = undefined;
        encodeNgramLE(window[0..cap_tokens], buf[0 .. cap_tokens * 4]);
        return stableHash(buf[0 .. cap_tokens * 4], self.seed);
    }

    fn baseRaw(self: *const Ranker, tokens: []const u32, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const types.RankedSegment, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        if (tokens.len == 0) return 0.0;

        const gram_limit = @min(self.num_ngrams, tokens.len);
        const rates = try allocator.alloc(f32, gram_limit);
        defer allocator.free(rates);

        var ngram_score: f32 = 0.0;
        var weight_sum: f32 = 0.0;
        var gram: usize = 1;
        while (gram <= gram_limit) : (gram += 1) {
            const weight_idx = @min(gram - 1, self.ngram_weights.len - 1);
            const w = self.ngram_weights[weight_idx];
            const cand_windows: usize = tokens.len - gram + 1;
            var ref_count: usize = retrieved.len;
            if (exclude_pos) |ep| {
                ref_count = 0;
                for (retrieved) |seg| {
                    if (seg.position == ep) continue;
                    ref_count += 1;
                }
            }
            const total_compare: usize = cand_windows * @max(ref_count, 1);

            var token_windows = std.AutoHashMap(u64, void).init(allocator);
            defer token_windows.deinit();

            var ti: usize = 0;
            while (ti + gram <= tokens.len) : (ti += 1) {
                try token_windows.put(self.windowHash(tokens[ti .. ti + gram]), {});
            }

            var matches_total: usize = 0;
            for (retrieved) |seg| {
                if (exclude_pos) |ep| {
                    if (seg.position == ep) continue;
                }
                const sim = seg.score;
                if (math.isNan(sim) or math.isInf(sim)) continue;
                var matches: usize = 0;
                var si: usize = 0;
                while (si + gram <= seg.tokens.len) : (si += 1) {
                    if (token_windows.contains(self.windowHash(seg.tokens[si .. si + gram]))) {
                        matches += 1;
                    }
                }
                matches_total += matches;
            }

            var rate: f32 = 0.0;
            if (total_compare > 0) {
                rate = @as(f32, @floatFromInt(matches_total)) / @as(f32, @floatFromInt(total_compare));
            }
            rates[gram - 1] = rate;
            weight_sum += w;
            ngram_score += w * rate;
        }

        const ngram_norm = if (weight_sum > 0.0) ngram_score / weight_sum else 0.0;
        const diversity_score = try self.computeTokenDiversity(tokens, allocator);
        const proximity = self.anchorProximity(candidate_pos, exclude_pos, retrieved);
        const raw = ngram_norm + RankerConfig.DIVERSITY_WEIGHT * diversity_score + RankerConfig.PROXIMITY_WEIGHT * proximity;

        if (grad_buf) |gb| {
            if (weight_sum > 0.0) {
                var gi: usize = 0;
                while (gi < gram_limit) : (gi += 1) {
                    gb[gi] = (rates[gi] - ngram_norm) / weight_sum;
                }
            }
            if (gram_limit < gb.len) {
                @memset(gb[gram_limit..], 0.0);
            }
        }

        return raw;
    }

    fn scoreSequenceRaw(self: *const Ranker, tokens: []const u32, ssi: *const SSI, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        return self.baseRaw(tokens, 0, null, retrieved, grad_buf, allocator);
    }

    fn scoreSequenceAlloc(self: *const Ranker, tokens: []const u32, ssi: *const SSI, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        const raw = try self.baseRaw(tokens, 0, null, retrieved, null, allocator);
        return sigmoidScale(raw);
    }

    pub fn scoreSequence(self: *const Ranker, tokens: []const u32, ssi: *const SSI) !f32 {
        return self.scoreSequenceAlloc(tokens, ssi, self.allocator);
    }

    fn combinedScore(self: *const Ranker, tokens: []const u32, query: []const u32, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const types.RankedSegment, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        const raw = try self.baseRaw(tokens, candidate_pos, exclude_pos, retrieved, grad_buf, allocator);
        const base_scaled = sigmoidScale(raw);
        if (query.len == 0) return base_scaled;

        var overlap: f32 = 0.0;
        var jaccard: f32 = 0.0;
        if (tokens.len > 0 and query.len > 0) {
            const m = try self.setMetrics(tokens, query, allocator);
            overlap = @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(@max(@min(m.card_a, m.card_b), 1)));
            const denom = m.card_a + m.card_b - m.intersection;
            jaccard = @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(@max(denom, 1)));
        } else if (tokens.len == 0 and query.len == 0) {
            jaccard = 1.0;
        }

        return math.clamp(base_scaled * RankerConfig.BASE_SCORE_WEIGHT + overlap * RankerConfig.OVERLAP_WEIGHT + jaccard * RankerConfig.JACCARD_WEIGHT, 0.0, 1.0);
    }

    fn scoreSequenceWithQueryAlloc(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(query, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        return self.combinedScore(tokens, query, 0, null, retrieved, null, allocator);
    }

    pub fn scoreSequenceWithQuery(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI) !f32 {
        return self.scoreSequenceWithQueryAlloc(tokens, query, ssi, self.allocator);
    }

    fn computeTokenDiversity(self: *const Ranker, tokens: []const u32, allocator: Allocator) !f32 {
        _ = self;
        if (tokens.len == 0) return 0.0;

        var unique_tokens = std.AutoHashMap(u32, void).init(allocator);
        defer unique_tokens.deinit();

        for (tokens) |token| {
            try unique_tokens.put(token, {});
        }

        const unique_count = unique_tokens.count();
        const diversity = @as(f32, @floatFromInt(unique_count)) / @as(f32, @floatFromInt(tokens.len));

        return diversity;
    }

    fn setMetrics(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !struct { intersection: usize, card_a: usize, card_b: usize } {
        _ = self;
        var set_a = std.AutoHashMap(u32, void).init(allocator);
        defer set_a.deinit();

        for (tokens) |token| {
            try set_a.put(token, {});
        }

        var set_b = std.AutoHashMap(u32, void).init(allocator);
        defer set_b.deinit();

        for (query) |qtoken| {
            try set_b.put(qtoken, {});
        }

        var intersection: usize = 0;
        var it = set_a.keyIterator();
        while (it.next()) |key| {
            if (set_b.contains(key.*)) {
                intersection += 1;
            }
        }

        return .{
            .intersection = intersection,
            .card_a = set_a.count(),
            .card_b = set_b.count(),
        };
    }

    fn computeTokenOverlap(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !f32 {
        if (tokens.len == 0 or query.len == 0) return 0.0;

        const m = try self.setMetrics(tokens, query, allocator);
        const denom = @min(m.card_a, m.card_b);
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(denom));
    }

    fn computeJaccardSimilarity(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !f32 {
        if (tokens.len == 0 and query.len == 0) return 1.0;
        if (tokens.len == 0 or query.len == 0) return 0.0;

        const m = try self.setMetrics(tokens, query, allocator);
        const denom = m.card_a + m.card_b - m.intersection;
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(denom));
    }

    fn anchorProximity(self: *const Ranker, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const types.RankedSegment) f32 {
        _ = self;
        if (retrieved.len == 0) return 0.0;

        var anchors: usize = 0;
        var total_dist: u64 = 0;
        for (retrieved) |seg| {
            if (exclude_pos) |ep| {
                if (seg.position == ep) continue;
            }
            if (!seg.anchor) continue;
            anchors += 1;
            const d = if (seg.position >= candidate_pos) seg.position - candidate_pos else candidate_pos - seg.position;
            total_dist += @min(d, RankerConfig.PROXIMITY_SCALE);
        }
        if (anchors == 0) return 0.0;
        const mean = total_dist / @as(u64, @intCast(anchors));
        return 1.0 - math.clamp(@as(f32, @floatFromInt(mean)) / @as(f32, @floatFromInt(RankerConfig.PROXIMITY_SCALE)), 0.0, 1.0);
    }

    pub fn rankCandidates(self: *const Ranker, candidates: []types.RankedSegment, ssi: *const SSI, allocator: Allocator) !void {
        return self.rankCandidatesWithQuery(candidates, &[_]u32{}, ssi, allocator);
    }

    fn rearrangeCandidatesByIndices(candidates: []types.RankedSegment, indices: []const usize, scores: []const f32, allocator: Allocator) !void {
        if (candidates.len == 0) return;
        if (indices.len != candidates.len) return error.LengthMismatch;
        if (scores.len != candidates.len) return error.LengthMismatch;

        const copy = try allocator.alloc(types.RankedSegment, candidates.len);
        defer allocator.free(copy);

        for (candidates, 0..) |c, idx| {
            copy[idx] = c;
        }

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const src = indices[i];
            candidates[i] = copy[src];
            candidates[i].score = scores[src];
        }
    }

    fn sortCandidatesByScore(candidates: []types.RankedSegment, scores: []const f32, allocator: Allocator) !void {
        if (candidates.len == 0) return;

        const indices = try allocator.alloc(usize, candidates.len);
        defer allocator.free(indices);

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            indices[i] = i;
        }

        const SortContext = struct {
            scores: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const score_a = ctx.scores[a];
                const score_b = ctx.scores[b];
                if (math.isNan(score_a)) return false;
                if (math.isNan(score_b)) return true;
                return score_a > score_b;
            }
        };
        std.mem.sort(usize, indices, SortContext{ .scores = scores }, SortContext.lessThan);

        try rearrangeCandidatesByIndices(candidates, indices, scores, allocator);
    }

    pub fn rankCandidatesWithQuery(self: *const Ranker, candidates: []types.RankedSegment, query: []const u32, ssi: *const SSI, allocator: Allocator) !void {
        if (candidates.len == 0) return;

        const scores = try allocator.alloc(f32, candidates.len);
        defer allocator.free(scores);

        if (query.len > 0) {
            const retrieved = try ssi.retrieveTopK(query, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
            defer freeRankedSegments(retrieved, allocator);
            var i: usize = 0;
            while (i < candidates.len) : (i += 1) {
                scores[i] = try self.combinedScore(candidates[i].tokens, query, candidates[i].position, candidates[i].position, retrieved, null, allocator);
            }
        } else {
            var i: usize = 0;
            while (i < candidates.len) : (i += 1) {
                const retrieved = try ssi.retrieveTopK(candidates[i].tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
                scores[i] = self.combinedScore(candidates[i].tokens, &.{}, candidates[i].position, candidates[i].position, retrieved, null, allocator) catch |err| {
                    freeRankedSegments(retrieved, allocator);
                    return err;
                };
                freeRankedSegments(retrieved, allocator);
            }
        }

        try sortCandidatesByScore(candidates, scores, allocator);
    }

    pub fn batchScore(self: *const Ranker, sequences: []const []const u32, ssi: *const SSI, allocator: Allocator) ![]f32 {
        if (sequences.len == 0) return allocator.alloc(f32, 0);

        const batch_size = sequences.len;
        const scores = try allocator.alloc(f32, batch_size);
        errdefer allocator.free(scores);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            scores[b] = try self.scoreSequenceAlloc(sequences[b], ssi, allocator);
        }
        return scores;
    }

    pub fn topKHeap(self: *const Ranker, ssi: *const SSI, query: []const u32, k: usize, allocator: Allocator) ![]types.RankedSegment {
        if (k == 0) return allocator.alloc(types.RankedSegment, 0);

        const retrieval_count = @max(k, RankerConfig.DEFAULT_TOP_N_RETRIEVAL);

        var heap = std.PriorityQueue(types.RankedSegment, void, RankedSegmentComparator.lessThan).init(allocator, {});
        defer {
            while (heap.removeOrNull()) |item| {
                var m = item;
                m.deinit(allocator);
            }
            heap.deinit();
        }

        const candidates = try ssi.retrieveTopK(query, retrieval_count, allocator);
        defer freeRankedSegments(candidates, allocator);

        const reference_len = @min(candidates.len, RankerConfig.SCORE_RETRIEVAL_LIMIT);
        const reference = candidates[0..reference_len];

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const cand = candidates[i];
            const score = try self.combinedScore(cand.tokens, query, cand.position, cand.position, reference, null, allocator);

            if (math.isNan(score) or math.isInf(score)) continue;

            if (heap.count() < k) {
                var ranked = try types.RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                heap.add(ranked) catch |err| {
                    ranked.deinit(allocator);
                    return err;
                };
            } else if (heap.peek()) |top| {
                if (math.isNan(top.score) or score > top.score) {
                    var removed = heap.remove();
                    removed.deinit(allocator);
                    var ranked = try types.RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                    heap.add(ranked) catch |err| {
                        ranked.deinit(allocator);
                        return err;
                    };
                }
            }
        }

        const result_count = heap.count();
        const top_n = try allocator.alloc(types.RankedSegment, result_count);
        var n_placed: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < n_placed) : (j += 1) {
                top_n[result_count - 1 - j].deinit(allocator);
            }
            allocator.free(top_n);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                top_n[idx] = item;
                n_placed += 1;
            } else {
                var mutable_item = item;
                mutable_item.deinit(allocator);
            }
        }

        return top_n;
    }

    pub fn updateWeights(self: *Ranker, gradients: []const f32) void {
        var i: usize = 0;
        while (i < @min(self.ngram_weights.len, gradients.len)) : (i += 1) {
            const grad = gradients[i];
            if (math.isNan(grad) or math.isInf(grad)) continue;
            self.ngram_weights[i] -= RankerConfig.LEARNING_RATE * grad;
            self.ngram_weights[i] = math.clamp(self.ngram_weights[i], 0.0, 1.0);
        }
    }

    pub fn minHashSignature(self: *const Ranker, tokens: []const u32) ![]u64 {
        if (tokens.len == 0) {
            const sig = try self.allocator.alloc(u64, self.num_hash_functions);
            @memset(sig, std.math.maxInt(u64));
            return sig;
        }

        const sig = try self.allocator.alloc(u64, self.num_hash_functions);
        errdefer self.allocator.free(sig);

        const token_hashes = try self.allocator.alloc(u64, tokens.len);
        defer self.allocator.free(token_hashes);

        var ti: usize = 0;
        while (ti < tokens.len) : (ti += 1) {
            const le_bytes = tokenToLEBytes(tokens[ti]);
            token_hashes[ti] = stableHash(&le_bytes, self.seed);
        }

        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            var min_hash: u64 = std.math.maxInt(u64);
            const seed_a = self.lsh_hash_params[h * 2];
            const seed_b = self.lsh_hash_params[h * 2 + 1];
            for (token_hashes) |th| {
                const hash_val = (th ^ seed_a) *% RankerConfig.HASH_SEED_MULTIPLIER_B +% seed_b;
                if (hash_val < min_hash) {
                    min_hash = hash_val;
                }
            }
            sig[h] = min_hash;
        }
        return sig;
    }

    pub fn jaccardSimilarityFromSignatures(sig1: []const u64, sig2: []const u64) f32 {
        if (sig1.len != sig2.len) return 0.0;
        if (sig1.len == 0) return 0.0;

        var matches: usize = 0;
        var i: usize = 0;
        while (i < sig1.len) : (i += 1) {
            if (sig1[i] == sig2[i]) {
                matches += 1;
            }
        }
        return @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(sig1.len));
    }

    pub fn minHashBitmask(self: *const Ranker, tokens: []const u32) ![]u64 {
        const words = (self.num_hash_functions + 63) / 64;
        const mask = try self.allocator.alloc(u64, words);
        errdefer self.allocator.free(mask);
        @memset(mask, 0);

        const sig = try self.minHashSignature(tokens);
        defer self.allocator.free(sig);

        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            if ((sig[h] & 1) != 0) {
                mask[h / 64] |= @as(u64, 1) << @intCast(h % 64);
            }
        }
        return mask;
    }

    pub fn jaccardFromBitmasks(mask1: []const u64, mask2: []const u64, valid_bits: usize) f32 {
        if (mask1.len != mask2.len) return 0.0;
        if (mask1.len == 0 or valid_bits == 0) return 0.0;

        const vector_len: usize = 4;
        const full_words = valid_bits / 64;
        var matches: usize = 0;
        var w: usize = 0;
        while (w + vector_len <= full_words) : (w += vector_len) {
            const a: @Vector(vector_len, u64) = mask1[w..][0..vector_len].*;
            const b: @Vector(vector_len, u64) = mask2[w..][0..vector_len].*;
            const agree = ~(a ^ b);
            var v: usize = 0;
            while (v < vector_len) : (v += 1) {
                matches += @popCount(agree[v]);
            }
        }
        while (w < full_words) : (w += 1) {
            matches += @popCount(~(mask1[w] ^ mask2[w]));
        }
        if (full_words < mask1.len) {
            const rem = valid_bits % 64;
            const valid_mask: u64 = if (rem == 0) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(rem)) - 1;
            matches += @popCount((~(mask1[full_words] ^ mask2[full_words])) & valid_mask);
        }
        const denom = valid_bits;
        if (denom == 0) return 0.0;
        const ratio = @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(denom));
        const estimate = 2.0 * ratio - 1.0;
        return math.clamp(estimate, 0.0, 1.0);
    }

    pub fn jaccardSignatureBitmask(self: *const Ranker, tokens1: []const u32, tokens2: []const u32) !f32 {
        const mask1 = try self.minHashBitmask(tokens1);
        defer self.allocator.free(mask1);
        const mask2 = try self.minHashBitmask(tokens2);
        defer self.allocator.free(mask2);
        return jaccardFromBitmasks(mask1, mask2, self.num_hash_functions);
    }

    pub fn estimateJaccard(set1: BitSet, set2: BitSet) f32 {
        const len1 = set1.bits.len;
        const len2 = set2.bits.len;
        const max_words = @max(len1, len2);

        if (max_words == 0) return 1.0;

        var intersect: usize = 0;
        var union_count: usize = 0;
        var i: usize = 0;
        while (i < max_words) : (i += 1) {
            const w1: u64 = if (i < len1) set1.bits[i] else 0;
            const w2: u64 = if (i < len2) set2.bits[i] else 0;
            intersect += @popCount(w1 & w2);
            union_count += @popCount(w1 | w2);
        }
        return if (union_count == 0) 1.0 else @as(f32, @floatFromInt(intersect)) / @as(f32, @floatFromInt(union_count));
    }

    pub fn vectorScore(embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        if (!mem.eql(usize, embedding.shape.dims, query_emb.shape.dims)) return Error.ShapeMismatch;
        if (embedding.data.len != query_emb.data.len) return Error.ShapeMismatch;
        if (embedding.data.len == 0) return 0.0;

        var dot_prod: f32 = 0.0;
        var norm_emb: f32 = 0.0;
        var norm_query: f32 = 0.0;

        const len = embedding.data.len;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const e = embedding.data[i];
            const q = query_emb.data[i];

            if (math.isNan(e) or math.isNan(q)) continue;
            if (math.isInf(e) or math.isInf(q)) continue;

            dot_prod += e * q;
            norm_emb += e * e;
            norm_query += q * q;
        }

        if (norm_emb <= 0.0 or norm_query <= 0.0) return 0.0;

        norm_emb = math.sqrt(norm_emb);
        norm_query = math.sqrt(norm_query);

        const result = dot_prod / (norm_emb * norm_query);
        return math.clamp(result, -1.0, 1.0);
    }

    pub fn dotProductScore(embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        if (!mem.eql(usize, embedding.shape.dims, query_emb.shape.dims)) return Error.ShapeMismatch;
        if (embedding.data.len != query_emb.data.len) return Error.ShapeMismatch;
        if (embedding.data.len == 0) return 0.0;

        var dot_prod: f32 = 0.0;
        const len = embedding.data.len;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const e = embedding.data[i];
            const q = query_emb.data[i];

            if (math.isNan(e) or math.isNan(q)) continue;
            if (math.isInf(e) or math.isInf(q)) continue;

            dot_prod += e * q;
        }
        return dot_prod;
    }

    pub fn weightedAverage(scores: []const f32, weights: []const f32) !f32 {
        if (scores.len != weights.len) return error.LengthMismatch;
        if (scores.len == 0) return 0.0;

        var num: f32 = 0.0;
        var den: f32 = 0.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            const w = weights[i];

            if (math.isNan(s) or math.isNan(w)) continue;
            if (math.isInf(w) or w < 0.0) return error.InvalidParameter;
            if (math.isInf(s)) continue;

            num += s * w;
            den += w;
        }

        if (den == 0.0) return 0.0;
        return num / den;
    }

    pub fn exponentialDecay(scores: []f32, decay_factor: f32) !void {
        if (scores.len == 0) return;
        if (decay_factor <= 0.0 or decay_factor >= 1.0) return error.InvalidParameter;

        var current_decay: f32 = 1.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                scores[i] *= current_decay;
            }
            current_decay *= decay_factor;
        }
    }

    pub fn normalizeScores(self: *const Ranker, scores: []f32) void {
        _ = self;
        normalizeScoresStatic(scores);
    }

    fn normalizeScoresStatic(scores: []f32) void {
        if (scores.len == 0) return;

        var min_score: f32 = math.inf(f32);
        var max_score: f32 = -math.inf(f32);
        var valid_count: usize = 0;

        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            if (math.isNan(s) or math.isInf(s)) continue;
            valid_count += 1;
            if (s < min_score) min_score = s;
            if (s > max_score) max_score = s;
        }

        if (valid_count == 0) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                scores[i] = 0.0;
            }
            return;
        }
        if (max_score == min_score) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                    scores[i] = 0.5;
                } else {
                    scores[i] = 0.0;
                }
            }
            return;
        }

        const range = max_score - min_score;
        i = 0;
        while (i < scores.len) : (i += 1) {
            if (math.isNan(scores[i]) or math.isInf(scores[i])) {
                scores[i] = 0.0;
            } else {
                scores[i] = (scores[i] - min_score) / range;
            }
        }
    }

    pub fn rankByMultipleCriteria(self: *const Ranker, candidates: []types.RankedSegment, criteria: []const []const f32, weights: []const f32, allocator: Allocator) !void {
        _ = self;
        if (candidates.len == 0) return;
        if (criteria.len == 0) return;
        if (weights.len == 0) return;

        const num_cand = candidates.len;
        const num_crit = @min(criteria.len, weights.len);

        var cr: usize = 0;
        while (cr < num_crit) : (cr += 1) {
            if (criteria[cr].len < num_cand) return error.LengthMismatch;
        }

        const combined = try allocator.alloc(f32, num_cand);
        defer allocator.free(combined);

        var c: usize = 0;
        while (c < num_cand) : (c += 1) {
            var crit_score: f32 = 0.0;
            cr = 0;
            while (cr < num_crit) : (cr += 1) {
                const score_val = criteria[cr][c];
                const weight_val = weights[cr];
                if (!math.isNan(score_val) and !math.isNan(weight_val) and !math.isInf(score_val) and !math.isInf(weight_val)) {
                    crit_score += score_val * weight_val;
                }
            }
            combined[c] = crit_score;
        }

        try sortCandidatesByScore(candidates, combined, allocator);
    }

    pub fn streamingRank(self: *const Ranker, reader: anytype, ssi: *const SSI, k: usize, allocator: Allocator) ![]types.RankedSegment {
        if (k == 0) return allocator.alloc(types.RankedSegment, 0);

        var rolling_buffer = std.ArrayList(u32).init(allocator);
        defer rolling_buffer.deinit();

        var heap = std.PriorityQueue(types.RankedSegment, void, RankedSegmentComparator.lessThan).init(allocator, {});
        defer {
            while (heap.removeOrNull()) |item| {
                var m = item;
                m.deinit(allocator);
            }
            heap.deinit();
        }

        var leftover_bytes: [3]u8 = undefined;
        var leftover_len: usize = 0;
        var position: u64 = 0;
        var read_buf: [RankerConfig.STREAMING_BUFFER_SIZE * @sizeOf(u32)]u8 = undefined;

        while (true) {
            const bytes_read = reader.read(&read_buf) catch |err| return err;
            if (bytes_read == 0) break;

            var combined_buf: []u8 = undefined;
            var combined_len: usize = 0;
            var combined_alloc: ?[]u8 = null;
            defer {
                if (combined_alloc) |ca| allocator.free(ca);
            }

            if (leftover_len > 0) {
                combined_len = leftover_len + bytes_read;
                combined_alloc = try allocator.alloc(u8, combined_len);
                combined_buf = combined_alloc.?;
                var ci: usize = 0;
                while (ci < leftover_len) : (ci += 1) {
                    combined_buf[ci] = leftover_bytes[ci];
                }
                var ri: usize = 0;
                while (ri < bytes_read) : (ri += 1) {
                    combined_buf[leftover_len + ri] = read_buf[ri];
                }
                leftover_len = 0;
            } else {
                combined_buf = read_buf[0..bytes_read];
                combined_len = bytes_read;
            }

            const full_tokens = combined_len / @sizeOf(u32);
            const remainder = combined_len % @sizeOf(u32);

            if (remainder > 0) {
                var ri: usize = 0;
                while (ri < remainder) : (ri += 1) {
                    leftover_bytes[ri] = combined_buf[full_tokens * @sizeOf(u32) + ri];
                }
                leftover_len = remainder;
            }

            var ti: usize = 0;
            while (ti < full_tokens) : (ti += 1) {
                const offset = ti * @sizeOf(u32);
                var token_bytes: [4]u8 = undefined;
                token_bytes[0] = combined_buf[offset + 0];
                token_bytes[1] = combined_buf[offset + 1];
                token_bytes[2] = combined_buf[offset + 2];
                token_bytes[3] = combined_buf[offset + 3];
                const token = mem.readInt(u32, &token_bytes, .little);
                try rolling_buffer.append(token);
            }

            while (rolling_buffer.items.len >= RankerConfig.STREAMING_WINDOW_SIZE) {
                const window = rolling_buffer.items[0..RankerConfig.STREAMING_WINDOW_SIZE];
                const score = try self.scoreSequenceAlloc(window, ssi, allocator);

                if (!math.isNan(score) and !math.isInf(score)) {
                    if (heap.count() < k) {
                        var seg = try types.RankedSegment.init(allocator, window, score, position, false);
                        heap.add(seg) catch |err| {
                            seg.deinit(allocator);
                            return err;
                        };
                    } else if (heap.peek()) |top| {
                        if (math.isNan(top.score) or score > top.score) {
                            var removed = heap.remove();
                            removed.deinit(allocator);
                            var seg = try types.RankedSegment.init(allocator, window, score, position, false);
                            heap.add(seg) catch |err| {
                                seg.deinit(allocator);
                                return err;
                            };
                        }
                    }
                }

                const shift = @min(rolling_buffer.items.len, RankerConfig.STREAMING_WINDOW_SIZE / 2);
                const remaining = rolling_buffer.items.len - shift;
                if (remaining > 0) {
                    std.mem.copyForwards(u32, rolling_buffer.items[0..remaining], rolling_buffer.items[shift..rolling_buffer.items.len]);
                }
                rolling_buffer.shrinkRetainingCapacity(remaining);
                position += shift;
            }
        }

        if (leftover_len > 0) return error.InvalidData;

        if (rolling_buffer.items.len > 0) {
            const tail = rolling_buffer.items;
            const score = try self.scoreSequenceAlloc(tail, ssi, allocator);
            if (!math.isNan(score) and !math.isInf(score)) {
                if (heap.count() < k) {
                    var seg = try types.RankedSegment.init(allocator, tail, score, position, false);
                    heap.add(seg) catch |err| {
                        seg.deinit(allocator);
                        return err;
                    };
                } else if (heap.peek()) |top| {
                    if (math.isNan(top.score) or score > top.score) {
                        var removed = heap.remove();
                        removed.deinit(allocator);
                        var seg = try types.RankedSegment.init(allocator, tail, score, position, false);
                        heap.add(seg) catch |err| {
                            seg.deinit(allocator);
                            return err;
                        };
                    }
                }
            }
        }

        const result_count = heap.count();
        const result = try allocator.alloc(types.RankedSegment, result_count);
        var n_placed: usize = 0;
        errdefer {
            var ei: usize = 0;
            while (ei < n_placed) : (ei += 1) {
                result[result_count - 1 - ei].deinit(allocator);
            }
            allocator.free(result);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                result[idx] = item;
                n_placed += 1;
            } else {
                var m = item;
                m.deinit(allocator);
            }
        }

        return result;
    }

    pub fn parallelScore(self: *const Ranker, sequences: []const []const u32, ssi: *const SSI, num_threads: usize) ![]f32 {
        if (sequences.len == 0) return self.allocator.alloc(f32, 0);

        const scores = try self.allocator.alloc(f32, sequences.len);
        errdefer self.allocator.free(scores);

        if (num_threads <= 1 or sequences.len <= 1) {
            var i: usize = 0;
            while (i < sequences.len) : (i += 1) {
                scores[i] = try self.scoreSequence(sequences[i], ssi);
            }
            return scores;
        }

        const cpu_count = std.Thread.getCpuCount() catch @as(usize, 1);
        const effective_threads = @min(@min(num_threads, sequences.len), cpu_count);
        const chunk_size = sequences.len / effective_threads;
        const remainder_count = sequences.len % effective_threads;

        const ThreadContext = struct {
            ranker: *const Ranker,
            seqs: []const []const u32,
            ssi_ptr: *const SSI,
            out: []f32,
            start: usize,
            end: usize,
            err_flag: bool,
        };

        const contexts = try self.allocator.alloc(ThreadContext, effective_threads);
        defer self.allocator.free(contexts);

        const threads = try self.allocator.alloc(std.Thread, effective_threads);
        defer self.allocator.free(threads);

        const thread_spawned = try self.allocator.alloc(bool, effective_threads);
        defer self.allocator.free(thread_spawned);
        @memset(thread_spawned, false);

        var offset: usize = 0;
        var t: usize = 0;
        while (t < effective_threads) : (t += 1) {
            const this_chunk = chunk_size + @as(usize, if (t < remainder_count) 1 else 0);
            contexts[t] = .{
                .ranker = self,
                .seqs = sequences,
                .ssi_ptr = ssi,
                .out = scores,
                .start = offset,
                .end = offset + this_chunk,
                .err_flag = false,
            };
            offset += this_chunk;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            threads[t] = std.Thread.spawn(.{}, struct {
                fn work(ctx: *ThreadContext) void {
                    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                    defer arena.deinit();
                    var si: usize = ctx.start;
                    while (si < ctx.end) : (si += 1) {
                        _ = arena.reset(.retain_capacity);
                        ctx.out[si] = ctx.ranker.scoreSequenceAlloc(ctx.seqs[si], ctx.ssi_ptr, arena.allocator()) catch {
                            ctx.err_flag = true;
                            return;
                        };
                    }
                }
            }.work, .{&contexts[t]}) catch {
                var si: usize = contexts[t].start;
                while (si < contexts[t].end) : (si += 1) {
                    scores[si] = self.scoreSequenceAlloc(sequences[si], ssi, self.allocator) catch {
                        contexts[t].err_flag = true;
                        break;
                    };
                }
                thread_spawned[t] = false;
                continue;
            };
            thread_spawned[t] = true;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (thread_spawned[t]) {
                threads[t].join();
            }
        }

        var had_error = false;
        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (contexts[t].err_flag) had_error = true;
        }

        if (had_error) {
            return error.ScoringFailed;
        }

        return scores;
    }

    pub fn calibrateWeights(self: *Ranker, training_data: []const []const u32, labels: []const f32, ssi: *const SSI, epochs: usize) !void {
        if (training_data.len == 0 or labels.len == 0) return error.InvalidParameter;
        if (training_data.len != labels.len) return error.LengthMismatch;

        const gradients = try self.allocator.alloc(f32, self.ngram_weights.len);
        defer self.allocator.free(gradients);

        const sample_grad = try self.allocator.alloc(f32, self.ngram_weights.len);
        defer self.allocator.free(sample_grad);

        var epoch: usize = 0;
        while (epoch < epochs) : (epoch += 1) {
            @memset(gradients, 0.0);

            var i: usize = 0;
            while (i < training_data.len) : (i += 1) {
                const sample = training_data[i];
                const label = labels[i];

                @memset(sample_grad, 0.0);
                const raw = try self.scoreSequenceRaw(sample, ssi, sample_grad, self.allocator);
                const pred = sigmoidScale(raw);

                if (math.isNan(pred) or math.isNan(label)) continue;
                if (math.isInf(pred) or math.isInf(label)) continue;

                const err_val = pred - label;
                const d_out = sigmoidDerivative(pred);
                const scale = err_val * d_out;

                var g: usize = 0;
                while (g < gradients.len) : (g += 1) {
                    gradients[g] += sample_grad[g] * scale;
                }
            }

            const n_samples: f32 = @floatFromInt(training_data.len);
            var g: usize = 0;
            while (g < gradients.len) : (g += 1) {
                gradients[g] = gradients[g] / n_samples;
            }

            self.updateWeights(gradients);
        }
    }

    pub fn exportModel(self: *const Ranker, path: []const u8) !void {
        const file = try core_io.createFilePath(path, .{});
        defer file.close();
        const writer = file.writer();
        try writer.writeInt(u8, 2, .little);
        try writer.writeInt(u64, @intCast(self.ngram_weights.len), .little);
        try writer.writeInt(u64, @intCast(self.num_ngrams), .little);
        var i: usize = 0;
        while (i < self.ngram_weights.len) : (i += 1) {
            const bits: u32 = @bitCast(self.ngram_weights[i]);
            try writer.writeInt(u32, bits, .little);
        }
        try writer.writeInt(u64, @intCast(self.num_hash_functions), .little);
        i = 0;
        while (i < self.lsh_hash_params.len) : (i += 1) {
            try writer.writeInt(u64, self.lsh_hash_params[i], .little);
        }
        try writer.writeInt(u64, self.seed, .little);
    }

    pub fn importModel(self: *Ranker, path: []const u8) !void {
        const file = try core_io.openFilePath(path, .{});
        defer file.close();
        const reader = file.reader();
        const version = try reader.readInt(u8, .little);
        if (version != 2) return error.InvalidVersion;

        const num_w = try reader.readInt(u64, .little);
        const num_ng = try reader.readInt(u64, .little);

        if (num_w == 0 or num_ng == 0) return error.InvalidParameter;
        if (num_w != num_ng) return error.InvalidParameter;
        if (num_w > std.math.maxInt(usize)) return error.InvalidParameter;
        if (num_w > RankerConfig.MAX_NGRAM_ORDER) return error.InvalidParameter;

        const num_w_usize: usize = @intCast(num_w);
        const num_ng_usize: usize = @intCast(num_ng);

        const new_weights = try self.allocator.alloc(f32, num_w_usize);
        errdefer self.allocator.free(new_weights);

        var i: usize = 0;
        while (i < num_w_usize) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            var weight: f32 = @bitCast(bits);
            if (math.isNan(weight) or math.isInf(weight)) weight = 0.0;
            new_weights[i] = weight;
        }

        const num_h = try reader.readInt(u64, .little);
        if (num_h == 0) return error.InvalidParameter;
        if (num_h > std.math.maxInt(usize) / 2) return error.InvalidParameter;

        const num_h_usize: usize = @intCast(num_h);

        const new_params = try self.allocator.alloc(u64, num_h_usize * 2);
        errdefer self.allocator.free(new_params);

        i = 0;
        while (i < new_params.len) : (i += 1) {
            new_params[i] = try reader.readInt(u64, .little);
        }
        const new_seed = try reader.readInt(u64, .little);

        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
        self.ngram_weights = new_weights;
        self.lsh_hash_params = new_params;
        self.num_ngrams = num_ng_usize;
        self.num_hash_functions = num_h_usize;
        self.seed = new_seed;
    }
};

test "Ranker score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3 }, 0, false);
    const score = try ranker.scoreSequence(&.{ 1, 2 }, &ssi);
    try testing.expect(score >= 0.0);
}

test "MinHash signature deterministic" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    try testing.expectEqualSlices(u64, sig1, sig2);
}

test "Jaccard similarity from signatures" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    const sim = Ranker.jaccardSimilarityFromSignatures(sig1, sig2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, @as(f32, 0.01));
}

test "Token diversity" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const div1 = try ranker.computeTokenDiversity(&.{ 1, 1, 1, 1 }, gpa);
    const div2 = try ranker.computeTokenDiversity(&.{ 1, 2, 3, 4 }, gpa);
    try testing.expect(div2 > div1);
}

test "Token overlap" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const overlap = try ranker.computeTokenOverlap(&.{ 1, 2, 3 }, &.{ 2, 3, 4 }, gpa);
    try testing.expect(overlap > 0.0 and overlap <= 1.0);
}

test "Estimate Jaccard" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var set1 = try BitSet.init(gpa, 128);
    defer set1.deinit();
    set1.set(0);
    set1.set(64);
    var set2 = try BitSet.init(gpa, 128);
    defer set2.deinit();
    set2.set(0);
    const est = Ranker.estimateJaccard(set1, set2);
    try testing.expect(est >= 0.0 and est <= 1.0);
}

test "Estimate Jaccard empty sets" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var set1 = try BitSet.init(gpa, 64);
    defer set1.deinit();
    var set2 = try BitSet.init(gpa, 64);
    defer set2.deinit();
    const est = Ranker.estimateJaccard(set1, set2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), est, 0.01);
}

test "Vector cosine score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 0.0;
    emb.data[2] = 0.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 0.0;
    qemb.data[2] = 0.0;
    const score = try Ranker.vectorScore(&emb, &qemb);
    try testing.expectApproxEqAbs(@as(f32, 1.0), score, @as(f32, 0.01));
}

test "Dot product score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 2.0;
    emb.data[2] = 3.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 2.0;
    qemb.data[2] = 3.0;
    const score = try Ranker.dotProductScore(&emb, &qemb);
    try testing.expectApproxEqAbs(@as(f32, 14.0), score, @as(f32, 0.01));
}

test "Weighted average" {
    const testing = std.testing;
    const scores = [_]f32{ 0.5, 0.8, 0.3 };
    const weights = [_]f32{ 1.0, 2.0, 1.0 };
    const avg = try Ranker.weightedAverage(&scores, &weights);
    try testing.expect(avg > 0.0 and avg < 1.0);
}

test "Weighted average rejects negative weights" {
    const testing = std.testing;
    const scores = [_]f32{ 0.5, 0.8 };
    const weights = [_]f32{ 1.0, -0.5 };
    try testing.expectError(error.InvalidParameter, Ranker.weightedAverage(&scores, &weights));
}

test "Exponential decay" {
    const testing = std.testing;
    var scores = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    try Ranker.exponentialDecay(&scores, 0.9);
    try testing.expect(scores[0] > scores[1]);
    try testing.expect(scores[1] > scores[2]);
    try testing.expect(scores[2] > scores[3]);
}

test "Normalize scores" {
    const testing = std.testing;
    var scores = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    Ranker.normalizeScoresStatic(&scores);
    try testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], @as(f32, 0.01));
    try testing.expectApproxEqAbs(@as(f32, 1.0), scores[3], @as(f32, 0.01));
}

test "MinHash bitmask signature identical sequences agree" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 128, 42);
    defer ranker.deinit();
    const mask1 = try ranker.minHashBitmask(&.{ 5, 6, 7, 8 });
    defer gpa.free(mask1);
    const mask2 = try ranker.minHashBitmask(&.{ 5, 6, 7, 8 });
    defer gpa.free(mask2);
    const sim = Ranker.jaccardFromBitmasks(mask1, mask2, 128);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, @as(f32, 0.01));
}

test "MinHash bitmask signature disjoint sequences below one" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 128, 42);
    defer ranker.deinit();
    const mask1 = try ranker.minHashBitmask(&.{ 1, 2, 3, 4 });
    defer gpa.free(mask1);
    const mask2 = try ranker.minHashBitmask(&.{ 100, 200, 300, 400 });
    defer gpa.free(mask2);
    const sim = Ranker.jaccardFromBitmasks(mask1, mask2, 128);
    try testing.expect(sim < 1.0);
    try testing.expect(sim >= 0.0);
}

test "Jaccard signature bitmask correlates with true overlap" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 256, 7);
    defer ranker.deinit();
    const sim_high = try ranker.jaccardSignatureBitmask(&.{ 1, 2, 3, 4, 5 }, &.{ 1, 2, 3, 4, 5, 6 });
    const sim_low = try ranker.jaccardSignatureBitmask(&.{ 1, 2, 3, 4, 5 }, &.{ 50, 60, 70, 80, 90 });
    try testing.expect(sim_high > sim_low);
}

test "Content scoring reflects indexed similarity" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3, 4, 5 }, 0, true);
    const matching = try ranker.scoreSequence(&.{ 1, 2, 3 }, &ssi);
    const unrelated = try ranker.scoreSequence(&.{ 900, 901, 902, 903, 904 }, &ssi);
    try testing.expect(matching > unrelated);
    try testing.expect(matching > 0.001);
}

test "Ngram order above limit rejected" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    try testing.expectError(error.InvalidParameter, Ranker.init(gpa, 65, 8, 42));
}

test "Model roundtrip preserves weights and params" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(path);
    const full = try std.fs.path.join(gpa, &.{ path, "m.bin" });
    defer gpa.free(full);
    try ranker.exportModel(full);

    var r2 = try Ranker.init(gpa, 1, 1, 1);
    defer r2.deinit();
    try r2.importModel(full);
    try testing.expectEqual(@as(usize, 4), r2.num_ngrams);
    try testing.expectEqual(@as(usize, 8), r2.num_hash_functions);
    try testing.expectEqual(@as(usize, 4), r2.ngram_weights.len);
    try testing.expectEqual(@as(usize, 16), r2.lsh_hash_params.len);
    try testing.expectEqual(@as(u64, 42), r2.seed);
    try testing.expectEqualSlices(f32, ranker.ngram_weights, r2.ngram_weights);
}

test "ImportModel truncated file errors and preserves state" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var r = try Ranker.init(gpa, 2, 4, 99);
    defer r.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(path);
    const full = try std.fs.path.join(gpa, &.{ path, "trunc.bin" });
    defer gpa.free(full);
    var bytes = std.ArrayList(u8).init(gpa);
    defer bytes.deinit();
    bytes.appendSlice(&[_]u8{2}) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u64, 3))) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u64, 3))) catch unreachable;
    const w0: u32 = @bitCast(@as(f32, 0.25));
    const w1: u32 = @bitCast(@as(f32, 0.50));
    const w2: u32 = @bitCast(@as(f32, 0.75));
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u32, w0))) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u32, w1))) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u32, w2))) catch unreachable;
    try tmp.dir.writeFile(.{ .sub_path = "trunc.bin", .data = bytes.items });
    try testing.expectError(error.EndOfStream, r.importModel(full));
    try testing.expectEqual(@as(usize, 2), r.num_ngrams);
    try testing.expectEqual(@as(usize, 4), r.num_hash_functions);
    try testing.expectEqual(@as(usize, 2), r.ngram_weights.len);
    try testing.expectEqual(@as(usize, 8), r.lsh_hash_params.len);
    try testing.expectEqual(@as(u64, 99), r.seed);
}

test "CalibrateWeights leaves weights unchanged when labels match predictions" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 2, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3, 4, 5 }, 0, true);
    const w0_before = ranker.ngram_weights[0];
    const w1_before = ranker.ngram_weights[1];
    const pred = try ranker.scoreSequence(&.{ 1, 2, 3 }, &ssi);
    const data = [_][]const u32{ &.{ 1, 2, 3 } };
    const labels = [_]f32{pred};
    try ranker.calibrateWeights(&data, &labels, &ssi, 1);
    try testing.expectApproxEqAbs(w0_before, ranker.ngram_weights[0], 1e-4);
    try testing.expectApproxEqAbs(w1_before, ranker.ngram_weights[1], 1e-4);
}

test "TopKHeap returns at most k results" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    const out = try ranker.topKHeap(&ssi, &.{1}, 3, gpa);
    defer {
        for (out) |*rs| rs.deinit(gpa);
        gpa.free(out);
    }
    try testing.expect(out.len <= 3);
}

test "StreamingRank returns at most k segments" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    var bytes = std.ArrayList(u8).init(gpa);
    defer bytes.deinit();
    var i: u32 = 0;
    while (i < 3000) : (i += 1) {
        const le = mem.toBytes(mem.nativeToLittle(u32, i));
        bytes.appendSlice(&le) catch unreachable;
    }
    var reader = std.io.fixedBufferStream(bytes.items);
    const out = try ranker.streamingRank(&reader.reader(), &ssi, 4, gpa);
    defer {
        for (out) |*rs| rs.deinit(gpa);
        gpa.free(out);
    }
    try testing.expect(out.len <= 4);
}

test "RankCandidatesWithQuery sorts by combined score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3, 4, 5 }, 0, true);
    const c1 = try types.RankedSegment.init(gpa, @constCast(&[_]u32{ 1, 2, 3 }), 0.0, 0, true);
    const c2 = try types.RankedSegment.init(gpa, @constCast(&[_]u32{ 900, 901, 902 }), 0.0, 1, false);
    var cands = [_]types.RankedSegment{ c1, c2 };
    defer {
        for (&cands) |*c| c.deinit(gpa);
    }
    try ranker.rankCandidatesWithQuery(&cands, &.{ 1, 2, 3 }, &ssi, gpa);
    try testing.expect(cands[0].score >= cands[1].score);
}
