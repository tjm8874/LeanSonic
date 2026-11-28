#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "CUDA driver: %s\n", message);
    std::exit(1);
}

template <typename T>
CUdeviceptr upload(const std::vector<T>& values) {
    CUdeviceptr ptr;
    check(cuMemAlloc(&ptr, values.size() * sizeof(T)));
    check(cuMemcpyHtoD(ptr, values.data(), values.size() * sizeof(T)));
    return ptr;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) return 2;
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) return 3;
    constexpr int width = 2304, channels = 1152, max_tokens = 255;
    constexpr int smem = 62976;
    std::array<CUmodule, 2> modules;
    std::array<CUfunction, 2> kernels;
    for (int arm = 0; arm < 2; ++arm) {
        check(cuModuleLoad(&modules[arm], argv[arm + 1]));
        check(cuModuleGetFunction(&kernels[arm], modules[arm], "ink2_moe_nvfp4_tc_up_situ_packed_scales"));
        check(cuFuncSetAttribute(kernels[arm], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    }
    std::mt19937 rng(417);
    auto bytes = [&](size_t count) {
        std::vector<uint8_t> data(count);
        for (auto& x : data) x = static_cast<uint8_t>(rng());
        return upload(data);
    };
    auto weights = bytes(size_t(2 * channels) * width / 2);
    auto ws = upload(std::vector<uint8_t>(size_t(2 * channels) * (width / 16) * 6 / 8 + 4, 0));
    auto amplitude = upload(std::vector<float>(128, 1.0f));
    auto acts = bytes(size_t(max_tokens) * width);
    auto xs = upload(std::vector<uint8_t>(size_t(max_tokens) * width / 8, 0x38));
    auto gain = upload(std::vector<float>(max_tokens, 1.0f));
    auto pairs = upload(std::vector<int>(max_tokens, 0));
    int begin = 0, end = 1;
    std::vector<int> counts(128, 0);
    auto routing = upload(counts);
    std::vector<int> shapes = {0,1,7,8,9,15,16,17,63,64,65,119,120,121,127,128,129,255};
    if (argc == 4) shapes = {9,129};
    size_t total = 0;
    for (int live : shapes) {
        counts[0] = live;
        check(cuMemcpyHtoD(routing, counts.data(), sizeof(int) * counts.size()));
        const size_t words = size_t(live + 2) * channels;
        std::vector<uint32_t> first(words), actual(words);
        CUdeviceptr buffer;
        check(cuMemAlloc(&buffer, words * sizeof(uint32_t)));
        CUdeviceptr output = buffer + channels * sizeof(uint32_t);
        void* args[] = {&weights,&ws,&amplitude,&acts,&xs,&gain,&routing,&pairs,&pairs,&begin,&end,&output};
        for (int arm = 0; arm < 2; ++arm) {
            check(cuMemsetD32(buffer, 0xffffffffu, words));
            check(cuLaunchKernel(kernels[arm], 9, (live+127)/128 + 1, 1, 512, 1, 1,
                                 smem, nullptr, args, nullptr));
            check(cuMemcpyDtoH(actual.data(), buffer, words * sizeof(uint32_t)));
            size_t nonzero = 0;
            for (size_t i = 0; i < words; ++i) {
                if (i < channels || i >= words-channels) {
                    if (actual[i] != 0xffffffffu) return 4;
                } else {
                    if ((actual[i] & 0x7f800000u) == 0x7f800000u) return 5;
                    nonzero += (actual[i] & 0x7fffffffu) != 0;
                }
            }
            if (live && nonzero < size_t(live)*channels/2) return 6;
            if (arm == 0) first = actual;
            else if (first != actual) return 7;
        }
        check(cuMemFree(buffer));
        total += size_t(live)*channels;
    }
    for (auto ptr : {weights,ws,amplitude,acts,xs,gain,pairs,routing}) check(cuMemFree(ptr));
    for (auto mod : modules) check(cuModuleUnload(mod));
    std::printf("%zu counts, %zu finite FP32 outputs and %zu guard words matched\n",
                shapes.size(),total,shapes.size()*2*channels);
}
