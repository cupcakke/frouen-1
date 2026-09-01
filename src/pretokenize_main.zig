const std = @import("std");
const MGT = @import("tokenizer/mgt.zig").MGT;
const dataset = @import("distributed/mmap_token_dataset.zig");

const CliError = error{
    InvalidArguments,
    InvalidInput,
    TokenOutOfRange,
    InputTooLarge,
};

fn openRead(path: []const u8) !std.fs.File {
    if (std.fs.path.isAbsolute(path)) return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
    return std.fs.cwd().openFile(path, .{ .mode = .read_only });
}

fn openWrite(path: []const u8) !std.fs.File {
    if (std.fs.path.isAbsolute(path)) return std.fs.createFileAbsolute(path, .{ .truncate = true, .read = true });
    return std.fs.cwd().createFile(path, .{ .truncate = true, .read = true });
}

fn extractText(allocator: std.mem.Allocator, line: []const u8) ![]const u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    if (parsed.value != .object) return CliError.InvalidInput;
    const value = parsed.value.object.get("text") orelse return CliError.InvalidInput;
    if (value != .string or value.string.len == 0) return CliError.InvalidInput;
    return try allocator.dupe(u8, value.string);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next();
    const input = args.next() orelse return CliError.InvalidArguments;
    const output = args.next() orelse return CliError.InvalidArguments;
    const vocab_path = args.next() orelse return CliError.InvalidArguments;
    const max_seq_raw = args.next() orelse return CliError.InvalidArguments;
    const max_seq = try std.fmt.parseInt(usize, max_seq_raw, 10);
    if (max_seq == 0) return CliError.InvalidArguments;
    const sample_count = if (args.next()) |raw| blk: {
        const value = try std.fmt.parseInt(usize, raw, 10);
        if (value == 0) return CliError.InvalidArguments;
        break :blk value;
    } else null;
    if (args.next() != null) return CliError.InvalidArguments;

    var tokenizer = try MGT.init(allocator, &.{}, &.{}, null, .english);
    defer tokenizer.deinit();
    try tokenizer.loadVocab(vocab_path);

    const pad_token_id: u32 = 0;
    if (tokenizer.next_token_id <= pad_token_id) return CliError.TokenOutOfRange;
    const window_length = std.math.add(usize, max_seq, 1) catch return CliError.InputTooLarge;
    const in_file = try openRead(input);
    defer in_file.close();
    var out_file = try openWrite(output);
    defer out_file.close();

    var payload = std.ArrayList(u32).init(allocator);
    defer payload.deinit();
    if (sample_count) |limit| {
        const expected_tokens = std.math.mul(usize, limit, window_length) catch return CliError.InputTooLarge;
        try payload.ensureTotalCapacity(expected_tokens);
    }

    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();
    try tokens.ensureTotalCapacity(window_length);

    var samples_written: usize = 0;
    var buffered = std.io.bufferedReader(in_file.reader());
    while (try buffered.reader().readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024 * 1024)) |raw_line| {
        defer allocator.free(raw_line);
        const line = std.mem.trimRight(u8, raw_line, "\r");
        if (line.len == 0) continue;
        if (sample_count) |limit| {
            if (samples_written >= limit) break;
        }
        const text = extractText(allocator, line) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => try allocator.dupe(u8, line),
        };
        defer allocator.free(text);

        tokens.clearRetainingCapacity();
        try tokenizer.encodeBounded(text, window_length, &tokens);
        var index: usize = 0;
        while (index < window_length) : (index += 1) {
            const token = if (index < tokens.items.len) tokens.items[index] else pad_token_id;
            if (token >= tokenizer.next_token_id) return CliError.TokenOutOfRange;
            try payload.append(token);
        }
        samples_written = std.math.add(usize, samples_written, 1) catch return CliError.InputTooLarge;
    }
    if (sample_count) |limit| {
        if (samples_written != limit) return CliError.InvalidInput;
    }

    const total = std.math.mul(u64, @intCast(samples_written), @intCast(window_length)) catch return CliError.InputTooLarge;
    var writer = out_file.writer();
    try writer.writeAll(dataset.magic);
    try writer.writeInt(u32, dataset.version, .little);
    try writer.writeInt(u32, dataset.token_type_u32, .little);
    try writer.writeInt(u64, dataset.header_size, .little);
    try writer.writeInt(u64, total, .little);
    try writer.writeInt(u64, tokenizer.next_token_id, .little);
    try writer.writeInt(u64, @intCast(max_seq), .little);
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, 0, .little);
    try writer.writeAll(std.mem.sliceAsBytes(payload.items));
}
