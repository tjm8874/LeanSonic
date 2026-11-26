// Build against the pinned llama.cpp revision in docs/inkling-reference.md.
// Read the original token IDs and score two single-sequence BF16 schedules.
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
constexpr int context = 512;
constexpr int vocab = 200058;
constexpr int first = context / 2;
constexpr int rows_per_chunk = context - first - 1;
constexpr int64_t row_bytes = 8 + 2 * int64_t(vocab);

void read_bytes(std::istream &input, void *dst, std::streamsize size) {
    if (!input.read(static_cast<char *>(dst), size)) {
        throw std::runtime_error("truncated oracle");
    }
}

struct Oracle {
    std::ifstream input;
    uint32_t chunks;
    int64_t rows_offset;

    explicit Oracle(const char *path) : input(path, std::ios::binary) {
        char magic[8];
        std::array<uint32_t, 3> dims;
        read_bytes(input, magic, sizeof(magic));
        read_bytes(input, dims.data(), sizeof(dims));
        if (std::string_view(magic, 8) != "_logits_" || dims[0] != context ||
            dims[1] != vocab || dims[2] == 0) {
            throw std::runtime_error("expected a little-endian Inkling 512-token oracle");
        }
        chunks = dims[2];
        rows_offset = 20 + int64_t(chunks) * context * 4;
        input.seekg(0, std::ios::end);
        if (input.tellg() != rows_offset + int64_t(chunks) * rows_per_chunk * row_bytes) {
            throw std::runtime_error("oracle dimensions and file length disagree");
        }
    }

    std::array<llama_token, context> tokens(uint32_t chunk) {
        std::array<llama_token, context> ids;
        input.seekg(20 + int64_t(chunk) * context * 4);
        read_bytes(input, ids.data(), sizeof(ids));
        if (std::any_of(ids.begin(), ids.end(), [](auto id) { return id < 0 || id >= vocab; })) {
            throw std::runtime_error("token outside Inkling vocabulary");
        }
        input.seekg(rows_offset + int64_t(chunk) * rows_per_chunk * row_bytes);
        return ids;
    }
};

void score(std::istream &ref, const float *x, uint32_t chunk, const char *schedule,
           int pos, std::vector<uint16_t> &codes) {
    float scale, floor;
    read_bytes(ref, &scale, sizeof(scale));
    read_bytes(ref, &floor, sizeof(floor));
    read_bytes(ref, codes.data(), vocab * sizeof(codes[0]));
    if (!x || !std::isfinite(scale) || scale < 0 || !std::isfinite(floor)) {
        throw std::runtime_error("missing logits or invalid reference row");
    }

    const int top = int(std::max_element(x, x + vocab) - x);
    double sum = 0;
    for (int i = 0; i < vocab; ++i) {
        sum += std::exp(x[i] - x[top]);
    }
    const float lse = x[top] + float(std::log(sum));
    double kl = 0;
    int ref_top = 0;
    float best = -INFINITY;
    for (int i = 0; i < vocab; ++i) {
        const float lp = scale * codes[i] + floor;
        if (lp > best) {
            ref_top = i;
            best = lp;
        }
        if (lp > -16.0f) {
            kl += std::exp(lp) * (lp - x[i] + lse);
        }
    }
    if (!std::isfinite(kl) || !std::isfinite(lse)) {
        throw std::runtime_error("non-finite score");
    }
    std::printf("%u,%s,%d,%.10g,%d,%d\n", chunk, schedule, pos, kl, top, ref_top);
}

void replay(llama_context *ctx, llama_batch_ext *batch, Oracle &ref, uint32_t chunk, bool stream) {
    const auto ids = ref.tokens(chunk);
    std::vector<uint16_t> codes(vocab);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
    const int end = stream ? context - 1 : context;
    for (int start = 0; start < end;) {
        const int count = stream ? (start < first ? 64 : 1) : context;
        llama_batch_ext_clear(batch);
        for (int i = 0; i < count; ++i) {
            const llama_pos pos = start + i;
            const int j = llama_batch_ext_add_token(batch, 0, ids[pos]);
            if (j < 0 || !llama_batch_ext_set_pos(batch, j, &pos) ||
                !llama_batch_ext_set_output_logits(batch, j, pos >= first)) {
                throw std::runtime_error("could not prepare batch");
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            throw std::runtime_error("llama_process failed");
        }
        for (int i = std::max(0, first - start); i < count && start + i < context - 1; ++i) {
            score(ref.input, llama_get_logits_ith(ctx, i), chunk, stream ? "stream" : "full",
                  start + i, codes);
        }
        start += count;
    }
}

int run(int argc, char **argv) {
    if (argc < 4) {
        throw std::runtime_error("usage: replay_inkling_bf16 MODEL.gguf ORACLE.kld CHUNK [CHUNK ...]");
    }
    Oracle oracle(argv[2]);
    std::vector<uint32_t> chunks;
    for (int i = 3; i < argc; ++i) {
        std::string_view arg(argv[i]);
        uint32_t chunk;
        auto result = std::from_chars(arg.data(), arg.data() + arg.size(), chunk);
        if (result.ec != std::errc() || result.ptr != arg.data() + arg.size() || chunk >= oracle.chunks) {
            throw std::runtime_error("invalid zero-based chunk number");
        }
        chunks.push_back(chunk);
    }

    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("failed to load model");
    const auto *v = llama_model_get_vocab(model.get());
    if (llama_vocab_n_tokens(v) != vocab || llama_vocab_get_add_bos(v)) {
        throw std::runtime_error("expected the pinned Inkling vocabulary without an added BOS");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = context;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("failed to initialize context");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("failed to allocate batch");

    std::puts("chunk,mode,pos,kld,top,ref_top");
    for (uint32_t chunk : chunks) {
        replay(ctx.get(), batch.get(), oracle, chunk, false);
        replay(ctx.get(), batch.get(), oracle, chunk, true);
    }
    if (std::fflush(stdout) != 0) throw std::runtime_error("failed writing CSV");
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    try {
        const int result = run(argc, argv);
        llama_backend_free();
        return result;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
