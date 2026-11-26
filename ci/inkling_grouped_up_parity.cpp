// Compare grouped NVFP4 up+SITU outputs from two Blackwell cubins.
// c++ -O2 -I/usr/local/cuda/include ci/inkling_grouped_up_parity.cpp -lcuda -o parity
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
    CUfunction kernels[2];
    for (int i = 0; i < 2; ++i) {
        check(cuModuleLoad(&modules[i], argv[i + 1]));
        check(cuModuleGetFunction(&kernels[i], modules[i], "ink2_moe_nvfp4_dp4a_grouped_up_situ"));
    }

    constexpr int experts = 2, tokens = 64, hidden = 2304, ff = 1152, groups = hidden / 16;
    std::mt19937 rng(20261126);
    auto packed_words = [&](size_t count) {
        std::vector<uint32_t> words(count);
        for (auto& word : words) word = rng();
        return upload(words);
    };
    auto weights = packed_words(experts * 2 * ff * hidden / 8);
    auto activations = packed_words(tokens * hidden / 4);
    std::vector<uint8_t> weight_scales(experts * 2 * ff * groups);
    for (auto& value : weight_scales) value = rng() % 123;
    auto ws = upload(weight_scales);
    std::vector<float> input_scales(tokens * groups);
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
    constexpr float sentinel = -98765.0f;
    std::vector<float> initial(experts * tokens * ff, sentinel);
    std::vector<float> outputs[2] = {initial, initial};
    auto output = upload(initial);
    const size_t bytes = initial.size() * sizeof(float);
    size_t live_values = 0;
    for (int n = 0; n <= tokens; ++n) {
        counts[0] = n;
        counts[1] = tokens - n;
        check(cuMemcpyHtoD(device_counts, counts.data(), counts.size() * sizeof(counts[0])));
        void* args[] = {&weights, &ws, &amax, &activations, &xs, &device_counts, &token_map, &pair_map, &output};
        for (int i = 0; i < 2; ++i) {
            check(cuMemcpyHtoD(output, initial.data(), bytes));
            check(cuLaunchKernel(kernels[i], 72, experts, 1, 256, 1, 1, 0, nullptr, args, nullptr));
            check(cuMemcpyDtoH(outputs[i].data(), output, bytes));
        }
        for (size_t i = 0; i < initial.size(); ++i) {
            const int pair = static_cast<int>(i / ff);
            const bool written = tokens - 1 - pair / experts < counts[pair % experts];
            const float actual = outputs[0][i];
            if (!std::isfinite(actual) || (actual != sentinel) != written ||
                std::memcmp(&outputs[0][i], &outputs[1][i], sizeof(float)) != 0) {
                std::fprintf(stderr, "count=%d index=%zu stock=%a candidate=%a\n", n, i, actual, outputs[1][i]);
                return 1;
            }
            live_values += written;
        }
    }
    for (auto ptr : {weights, ws, amax, activations, xs, device_counts, token_map, pair_map, output}) {
        check(cuMemFree(ptr));
    }
    for (auto module : modules) check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(gpu));
    std::printf("%zu active FP32 values matched across 65 occupancies; inactive rows untouched\n", live_values);
}
