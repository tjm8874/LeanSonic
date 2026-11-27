// Build against the llama.cpp revision pinned in docs/inkling-reference.md.
#include "ggml-backend.h"
#include "llama.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>



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
    if (argc != 5) throw std::runtime_error("usage: schedule-bf16 MODEL.gguf README.ids OUTPUT.f32 N_GPU_LAYERS");
    auto ids = read_ids(argv[2], 1024);
    std::ofstream output(argv[3], std::ios::binary);
    if (!output) throw std::runtime_error("cannot open output");
    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = std::stoi(argv[4]);
    auto *model = llama_model_load_from_file(argv[1], model_params);
    if (!model) throw std::runtime_error("model load failed");
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = params.n_ubatch = 256;
    params.n_seq_max = 1;
    params.n_threads = params.n_threads_batch = 2;
    auto *ctx = llama_init_from_model(model, params);
    if (!ctx) throw std::runtime_error("context initialization failed");
    auto *batch = llama_batch_ext_init(ctx);
    if (!batch) throw std::runtime_error("batch initialization failed");
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

    };

    for (int slot : {31, 32, 33, 34, 39, 44, 50, 57, 59, 63}) {
        llama_memory_clear(llama_get_memory(ctx), true);
        int length = 32 + slot;
        eval(ids.data() + 3*slot, length, 0);
        for (int step = 0; step < 2; ++step) {
            eval(ids.data() + 512 + 2*slot + step, 1, length + step);
            const float *row = llama_get_logits_ith(ctx, -1);
            if (!row || !std::all_of(row, row+200058, [](float v){ return std::isfinite(v); }))
                throw std::runtime_error("invalid BF16 logits");
            output.write(reinterpret_cast<const char *>(row), 200058 * sizeof(float));
            if (!output) throw std::runtime_error("output write failed");
        }
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
