// c++ -O2 -I/usr/local/cuda/include kernels/tests/inkling_expert_pages.cpp -lcuda -o pages
// ./pages candidate.cubin [reference.cubin]
#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <cstdlib>
#include <vector>

static void check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "CUDA error %d: %s\n", result, message ? message : "unknown");
    std::exit(1);
}

template<class T>
static CUdeviceptr upload(const std::vector<T>& data) {
    CUdeviceptr ptr;
    check(cuMemAlloc(&ptr, data.size() * sizeof(T)));
    check(cuMemcpyHtoD(ptr, data.data(), data.size() * sizeof(T)));
    return ptr;
}

static void compare(CUdeviceptr ptr, const std::vector<int>& expected, const char* name, int batch) {
    std::vector<int> actual(expected.size());
    check(cuMemcpyDtoH(actual.data(), ptr, actual.size() * sizeof(int)));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s batch=%d offset=%zu expected=%d actual=%d\n",
                         name, batch, i, expected[i], actual[i]);
            std::exit(1);
        }
    }
}

static void test(CUfunction group, int batch, bool concentrated) {
    constexpr int experts = 128, page = 64, top_k = 8;
    std::vector<int> ids(batch * top_k), counts(experts), pairs((batch + 63) / page * experts * page, -1);
    auto tokens = pairs;
    auto d_pairs = upload(pairs), d_tokens = upload(tokens);
    auto d_counts = upload(std::vector<int>(experts, -1));
    for (int token = 0; token < batch; ++token) {
        for (int k = 0; k < top_k; ++k) {
            int pair = token * top_k + k;
            // One pattern crosses every page for the highest-ID experts. The other
            // mixes hot and rare experts, leaving partially populated pages.
            int expert = concentrated ? 127 - k : (k < 3 ? 125 + k : (token * 17 + k * 19) % 125);
            ids[pair] = expert;
            int rank = counts[expert]++;
            int offset = (rank / page * experts + expert) * page + rank % page;
            pairs[offset] = pair;
            tokens[offset] = token;
        }
    }
    auto d_ids = upload(ids);
    void* args[] = {&d_ids, &d_counts, &d_tokens, &d_pairs, &batch};
    check(cuLaunchKernel(group, 32, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    compare(d_counts, counts, "counts", batch);
    compare(d_tokens, tokens, "token indices", batch);
    compare(d_pairs, pairs, "pair indices", batch);
    for (auto ptr : {d_ids, d_counts, d_tokens, d_pairs}) check(cuMemFree(ptr));
}

// One CTA per expert computes the first output tile. The reference runs its
// original 64-entry map on each page. The candidate consumes all five pages.
static void projections(CUmodule original, CUmodule candidate, bool nvfp4, bool up) {
    constexpr int batch = 257, experts = 2, hidden = 2304, ff = 1152;
    constexpr int page = 64, page_stride = 128 * page, pages = (batch + page - 1) / page;
    const int counts[] = {batch, 130};
    const int cols = up ? hidden : ff;
    const int stride = up ? ff : hidden;
    const char* name = nvfp4
        ? (up ? "ink2_moe_nvfp4_dp4a_grouped_up_situ" : "ink2_moe_nvfp4_dp4a_grouped_down")
        : (up ? "ink2_moe_mxfp8_grouped_up_situ" : "ink2_moe_mxfp8_grouped_down");
    std::mt19937 rng(42);
    std::vector<uint8_t> weight(experts * hidden * cols / (nvfp4 ? 2 : 1));
    for (auto& w : weight) w = nvfp4 ? rng() : rng() % 111;
    std::vector<uint8_t> scales(experts * hidden * cols / (nvfp4 ? 16 : 32));
    for (auto& x : scales) x = nvfp4 ? 29 + rng() % 23 : 119 + rng() % 4;
    auto w = upload(weight), ws = upload(scales);
    auto amax = upload(std::vector<float>{2.5f, 1.25f});
    const int input_rows = up ? batch : experts * batch;
    std::vector<float> floats(input_rows * cols);
    std::vector<int8_t> codes(input_rows * cols);
    for (auto& x : floats) x = (int(rng() % 2049) - 1024) / 1024.0f;
    for (auto& x : codes) x = int(rng() % 255) - 127;
    auto x = nvfp4 ? upload(codes) : upload(floats);
    auto xs = upload(std::vector<float>(input_rows * cols / 16, 0.01f));
    std::vector<int> tokens(pages * page_stride, -1), pairs(tokens);
    for (int ex = 0; ex < experts; ++ex) {
        for (int rank = 0; rank < counts[ex]; ++rank) {
            int offset = (rank / page * 128 + ex) * page + rank % page;
            tokens[offset] = (rank * 5 + ex) % batch;
            pairs[offset] = (batch - 1 - rank) * experts + ex;
        }
    }
    auto token_map = upload(tokens), pair_map = upload(pairs);
    auto d_counts = upload(std::vector<int>(experts));
    constexpr float sentinel = -98765.0f;
    std::vector<float> empty(batch * experts * stride, sentinel), expected(empty), actual(empty);
    const auto bytes = empty.size() * sizeof(float);
    auto out = upload(empty);
    for (bool wide : {false, true}) {
        CUfunction kernel;
        check(cuModuleGetFunction(&kernel, wide ? candidate : original, name));
        check(cuMemcpyHtoD(out, empty.data(), bytes));
        for (int p = 0; p < (wide ? 1 : pages); ++p) {
            int n[experts];
            for (int ex = 0; ex < experts; ++ex)
                n[ex] = wide ? counts[ex] : std::min(page, std::max(0, counts[ex] - p * page));
            check(cuMemcpyHtoD(d_counts, n, sizeof(n)));
            auto tm = token_map + p * page_stride * sizeof(int);
            auto pm = pair_map + p * page_stride * sizeof(int);
            std::vector<void*> args{&w, &ws};
            if (nvfp4) args.push_back(&amax);
            args.push_back(&x);
            if (nvfp4) args.push_back(&xs);
            args.push_back(&d_counts);
            if (up) args.push_back(&tm);
            args.push_back(&pm);
            args.push_back(&out);
            check(cuLaunchKernel(kernel, 1, experts, 1, 256, 1, 1, 0, nullptr, args.data(), nullptr));
        }
        check(cuMemcpyDtoH((wide ? actual : expected).data(), out, bytes));
    }
    for (size_t i = 0; i < empty.size(); ++i) {
        int pair = int(i / stride);
        bool live = batch - 1 - pair / experts < counts[pair % experts] && int(i % stride) < (up ? 16 : 32);
        if ((expected[i] != sentinel) != live || !std::isfinite(expected[i]) ||
            std::memcmp(&actual[i], &expected[i], sizeof(float))) {
            std::fprintf(stderr, "%s index=%zu reference=%a actual=%a\n", name, i, expected[i], actual[i]);
            std::exit(1);
        }
    }
    for (auto ptr : {w, ws, amax, x, xs, token_map, pair_map, d_counts, out}) check(cuMemFree(ptr));
    std::printf("%s: five pages matched reference; padding preserved\n", name);
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    check(cuInit(0));
    CUdevice gpu;
    CUcontext context;
    check(cuDeviceGet(&gpu, 0));
    check(cuDevicePrimaryCtxRetain(&context, gpu));
    check(cuCtxSetCurrent(context));
    CUmodule module;
    CUfunction group;
    check(cuModuleLoad(&module, argv[1]));
    check(cuModuleGetFunction(&group, module, "ink2_build_expert_groups_wide_b"));
    for (int batch : {1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024}) {
        for (bool concentrated : {false, true}) test(group, batch, concentrated);
    }
    if (argc == 3) {
        CUmodule original;
        check(cuModuleLoad(&original, argv[2]));
        for (bool nvfp4 : {false, true})
            for (bool up : {false, true}) projections(original, module, nvfp4, up);
        check(cuModuleUnload(original));
    }
    check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(gpu));
    std::puts("30 stable grouping and padding cases passed");
}
