// Build with the revision of llama.cpp recorded in docs/inkling-reference.md.
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
constexpr int vocab = 200058;
constexpr int context = 512;

std::vector<llama_token> read_ids(const fs::path &path) {
    std::ifstream input(path);
    std::vector<llama_token> ids;
    for (llama_token id; input >> id;) {
        if (id < 0 || id >= vocab) throw std::runtime_error("invalid token ID");
        ids.push_back(id);
    }
    if (!input.eof() || ids.empty()) throw std::runtime_error("invalid token fixture " + path.string());
    return ids;
}

void run(int argc, char **argv) {
    if (argc < 4) throw std::runtime_error("expected MODEL.gguf FIXTURE_DIR CASE ...");
    if (const char *directory = std::getenv("LEANSONIC_LLAMA_BACKEND_DIR")) {
        ggml_backend_load_all_from_path(directory);
    } else {
        ggml_backend_load_all();
    }
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
        throw std::runtime_error("missing GPU backend");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model || llama_vocab_n_tokens(llama_model_get_vocab(model.get())) != vocab)
        throw std::runtime_error("unexpected model");
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = context;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("failed to allocate context");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("failed to allocate batch");
    for (int arg = 3; arg < argc; ++arg) {
        const auto dir = fs::path(argv[2]) / argv[arg];
        const auto prefix = read_ids(dir / "prefix.ids");
        const auto targets = read_ids(dir / "answer.ids");
        if (prefix.size() + targets.size() > context) throw std::runtime_error("fixture exceeds context");
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        llama_synchronize(ctx.get());
        const auto output = std::string(argv[arg]) + ".f32";
        if (fs::exists(output)) throw std::runtime_error("output exists: " + output);
        std::ofstream out(output, std::ios::binary);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        int pos = 0;
        for (size_t row = 0; row < targets.size(); ++row) {
            llama_batch_ext_clear(batch.get());
            const int count = row == 0 ? static_cast<int>(prefix.size()) : 1;
            for (int i = 0; i < count; ++i, ++pos) {
                const auto id = row == 0 ? prefix[i] : targets[row - 1];
                const int j = llama_batch_ext_add_token(batch.get(), 0, id);
                if (j < 0 || !llama_batch_ext_set_pos(batch.get(), j, &pos) ||
                    !llama_batch_ext_set_output_logits(batch.get(), j, i == count - 1))
                    throw std::runtime_error("failed to prepare batch");
            }
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0)
                throw std::runtime_error("llama_process failed");
            const float *values = llama_get_logits_ith(ctx.get(), count - 1);
            if (!values) throw std::runtime_error("missing row");
            out.write(reinterpret_cast<const char *>(values), vocab * sizeof(float));
        }
        out.close();
    }
}

int main(int argc, char **argv) {
    try {
        run(argc, argv);
        llama_backend_free();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
