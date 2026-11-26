// Compare dense Q8 and grouped NVFP4 projection outputs from two Blackwell cubins.
// c++ -O2 -I/usr/local/cuda/include kernels/tests/inkling_mma_parity.cpp -lcuda -o parity
// ./parity stock.cubin candidate.cubin
#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        std::fprintf(stderr, "CUDA: %s\n", message ? message : "unknown error");
        std::exit(1);
    }
}

template <class T>
static CUdeviceptr upload(const std::vector<T>& values) {
    CUdeviceptr device;
    const auto bytes = values.size() * sizeof(T);
    check(cuMemAlloc(&device, bytes));
    check(cuMemcpyHtoD(device, values.data(), bytes));
    return device;
}

static void check_dense(const CUmodule (&modules)[2]) {
    CUfunction kernels[2];
    for (int i = 0; i < 2; ++i) {
        check(cuModuleGetFunction(&kernels[i], modules[i], "ink2_q8_dp4a_gemm_b"));
    }
    std::mt19937 rng(20261127);
    auto words = [&](size_t count) {
        std::vector<uint32_t> values(count);
        for (auto& value : values) value = rng();
        return upload(values);
    };
    auto scales = [&](size_t count) {
        std::vector<float> values(count);
        for (auto& value : values) value = (1u + rng() % 113) * 0.0004f;
        return upload(values);
    };
    size_t live_values = 0;
    auto compare_shape = [&](int rows, int cols, int tokens) {
        auto w = words(rows * cols / 4);
        auto ws = scales(rows * cols / 64);
        auto x = words(tokens * cols / 4);
        auto xs = scales(tokens * cols / 64);
        constexpr float sentinel = -54321.0f;
        int stride = rows + 3;
        std::vector<float> empty(stride * tokens, sentinel);
        std::vector<float> result[2] = {empty, empty};
        auto output = upload(empty);
        const size_t bytes = empty.size() * sizeof(float);
        for (float cap : {0.0f, 10.0f}) {
            void* args[] = {&w, &ws, &x, &xs, &output, &rows, &cols, &tokens, &stride, &cap};
            for (int i = 0; i < 2; ++i) {
                check(cuMemcpyHtoD(output, empty.data(), bytes));
                check(cuLaunchKernel(kernels[i], (rows + 15) / 16, (tokens + 7) / 8, 1,
                                     256, 1, 1, 0, nullptr, args, nullptr));
                check(cuMemcpyDtoH(result[i].data(), output, bytes));
            }
            for (size_t i = 0; i < empty.size(); ++i) {
                const bool live = static_cast<int>(i % stride) < rows;
                const float reference = result[0][i];
                if (!std::isfinite(reference) || (reference != sentinel) != live ||
                    std::memcmp(&result[0][i], &result[1][i], sizeof(float)) != 0) {
                    std::fprintf(stderr, "Q8 GEMM rows=%d cols=%d tokens=%d cap=%g index=%zu: %a vs %a\n",
                                 rows, cols, tokens, cap, i, reference, result[1][i]);
                    std::exit(1);
                }
                live_values += live;
            }
        }
        for (auto ptr : {w, ws, x, xs, output}) check(cuMemFree(ptr));
    };
    for (int cols : {64, 128, 1024, 2304, 3072}) {
        for (int rows : {1, 15, 16, 17, 48, 65}) {
            for (int tokens : {1, 7, 8, 9, 63, 64}) compare_shape(rows, cols, tokens);
        }
    }
    std::printf("Q8 GEMM: %zu active FP32 values matched across 360 shapes/caps; padding untouched\n",
                live_values);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s stock.cubin candidate.cubin\n", argv[0]);
        return 2;
    }
    CUdevice gpu;
    CUcontext context;
    check(cuInit(0));
    check(cuDeviceGet(&gpu, 0));
    check(cuDevicePrimaryCtxRetain(&context, gpu));
    check(cuCtxSetCurrent(context));
    CUmodule modules[2];
    for (int i = 0; i < 2; ++i) {
        check(cuModuleLoad(&modules[i], argv[i + 1]));
    }

    constexpr int experts = 2, tokens = 64, hidden = 2304, ff = 1152, groups = hidden / 16;
    std::mt19937 rng(20261126);
    auto packed_words = [&](size_t count) {
        std::vector<uint32_t> words(count);
        for (auto& word : words) word = rng();
        return upload(words);
    };
    auto weights = packed_words(experts * 2 * ff * hidden / 8);
    auto activations = packed_words(experts * tokens * hidden / 4);
    std::vector<uint8_t> weight_scales(experts * 2 * ff * groups);
    for (auto& value : weight_scales) value = rng() % 123;
    auto ws = upload(weight_scales);
    std::vector<float> input_scales(experts * tokens * groups);
    for (auto& value : input_scales) value = (1u + rng() % 997) * 0.000003f;
    auto xs = upload(input_scales);
    auto amax = upload(std::vector<float>{2.5f, 1.25f});

    std::vector<int32_t> token_ids(experts * tokens), pair_ids(experts * tokens);
    for (int i = 0; i < experts * tokens; ++i) {
        token_ids[i] = (5 * i + i / tokens) % tokens;
        pair_ids[i] = (tokens - 1 - i % tokens) * experts + i / tokens;
    }
    auto token_map = upload(token_ids);
    auto pair_map = upload(pair_ids);
    std::vector<int32_t> counts(experts);
    auto device_counts = upload(counts);
    for (bool up : {true, false}) {
        const int stride = up ? ff : hidden;
        const char* name = up ? "ink2_moe_nvfp4_dp4a_grouped_up_situ" : "ink2_moe_nvfp4_dp4a_grouped_down";
        CUfunction kernels[2];
        for (int i = 0; i < 2; ++i) check(cuModuleGetFunction(&kernels[i], modules[i], name));
        constexpr float sentinel = -98765.0f;
        std::vector<float> initial(experts * tokens * stride, sentinel);
        std::vector<float> outputs[2] = {initial, initial};
        auto output = upload(initial);
        const size_t bytes = initial.size() * sizeof(float);
        size_t live_values = 0;
        for (int n = 0; n <= tokens; ++n) {
            counts[0] = n;
            counts[1] = tokens - n;
            check(cuMemcpyHtoD(device_counts, counts.data(), counts.size() * sizeof(counts[0])));
            void* up_args[] = {&weights, &ws, &amax, &activations, &xs, &device_counts, &token_map, &pair_map, &output};
            void* down_args[] = {&weights, &ws, &amax, &activations, &xs, &device_counts, &pair_map, &output};
            for (int i = 0; i < 2; ++i) {
                check(cuMemcpyHtoD(output, initial.data(), bytes));
                check(cuLaunchKernel(kernels[i], 72, experts, 1, 256, 1, 1, 0, nullptr, up ? up_args : down_args, nullptr));
                check(cuMemcpyDtoH(outputs[i].data(), output, bytes));
            }
            for (size_t i = 0; i < initial.size(); ++i) {
                const int pair = static_cast<int>(i / stride);
                const bool written = tokens - 1 - pair / experts < counts[pair % experts];
                const float actual = outputs[0][i];
                if (!std::isfinite(actual) || (actual != sentinel) != written ||
                    std::memcmp(&outputs[0][i], &outputs[1][i], sizeof(float)) != 0) {
                    std::fprintf(stderr, "%s count=%d index=%zu stock=%a candidate=%a\n", name, n, i, actual, outputs[1][i]);
                    return 1;
                }
                live_values += written;
            }
        }
        check(cuMemFree(output));
        std::printf("%s: %zu active FP32 values matched across 65 occupancies; inactive rows untouched\n", name, live_values);
    }
    for (auto ptr : {weights, ws, amax, activations, xs, device_counts, token_map, pair_map}) {
        check(cuMemFree(ptr));
    }
    check_dense(modules);
    for (auto module : modules) check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(gpu));
}
