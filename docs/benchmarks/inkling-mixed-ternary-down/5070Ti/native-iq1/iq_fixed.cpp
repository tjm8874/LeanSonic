// Build against the llama.cpp revision pinned in docs/inkling-reference.md.
#include "ggml-backend.h"
#include "llama.h"
#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

static std::vector<llama_token> read_ids(const char *path, size_t expected) {
    std::ifstream file(path);
    std::vector<llama_token> ids;
    std::string entry;
    while (std::getline(file, entry, ',')) {
        size_t end = 0;
        int id = std::stoi(entry, &end);
        if (id < 0 || id >= 200058 || entry.find_first_not_of(" \t\r\n", end) != std::string::npos)
            throw std::runtime_error("invalid Inkling token");
        ids.push_back(id);
    }
    if (ids.size() != expected) throw std::runtime_error("unexpected token count");
    return ids;
}

int main(int argc, char **argv) try {
    if (argc != 4) throw std::runtime_error("usage: benchmark_inkling_bf16 MODEL.gguf PROMPT.ids CONTINUATION.ids");
    auto prompt = read_ids(argv[2], 1024), continuation = read_ids(argv[3], 32);
    // This benchmark runs outside the publisher's release directory.
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&ggml_backend_load_all_from_path), &info) || !info.dli_fname)
        throw std::runtime_error("could not locate the pinned ggml library");
    const auto plugins = std::filesystem::canonical(info.dli_fname).parent_path();
    ggml_backend_load_all_from_path(plugins.c_str());
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
        throw std::runtime_error("no GPU backend in the pinned release");
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], model_params);
    if (!model) throw std::runtime_error("model load failed");
    auto params = llama_context_default_params();
    params.n_ctx = 8192;
    params.n_batch = params.n_ubatch = 1024;
    params.n_seq_max = 1;
    params.n_threads = params.n_threads_batch = 8;
    auto *ctx = llama_init_from_model(model, params);
    if (!ctx) throw std::runtime_error("context initialization failed");
    auto *batch = llama_batch_ext_init(ctx);
    auto *greedy = llama_sampler_init_greedy();
    if (!batch || !greedy) throw std::runtime_error("batch or sampler initialization failed");
    auto eval = [&](const llama_token *tokens, int count, int position) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < count; ++i) {
            int row = llama_batch_ext_add_token(batch, 0, tokens[i]);
            llama_pos pos = position + i;
            if (row < 0) throw std::runtime_error("batch append failed");
            llama_batch_ext_set_pos(batch, row, &pos);
        }
        llama_batch_ext_set_output_logits(batch, count - 1, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0)
            throw std::runtime_error("llama_process failed");
        return llama_sampler_sample(greedy, ctx, -1);
    };

    for (int run = 0; run < 5; ++run) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_synchronize(ctx);
        auto start = Clock::now();
        auto first = eval(prompt.data(), int(prompt.size()), 0);
        auto prefill_end = Clock::now();
        std::vector<llama_token> predictions; predictions.reserve(continuation.size());
        int matches = 0;
        auto next = first;
        for (int i = 0; i < int(continuation.size()); ++i) {
            predictions.push_back(next);
            matches += next == continuation[i];
            next = eval(&continuation[i], 1, int(prompt.size()) + i);
        }
        auto end = Clock::now();
        std::printf("%d %.4f %.4f %d %d\n", run, Milliseconds(prefill_end-start).count(),
                    Milliseconds(end-prefill_end).count(), matches, int(first));
        std::printf("P %d",run);
        for (auto id : predictions) std::printf(" %d",int(id));
        std::printf("\n");
    }
    llama_sampler_free(greedy);
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
