// Compare the original FP16 attention+Q8 kernel with the decode and speculative verification specialization.
// c++ -O2 -I/usr/local/cuda/include kernels/tests/inkling_decode_attention.cpp -lcuda -o parity
// ./parity reference.cubin candidate.cubin
#include <cuda.h>

#include <algorithm>
#include <array>
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

template<class T>
static CUdeviceptr upload(const std::vector<T>& values) {
    CUdeviceptr p;
    const size_t bytes = values.size() * sizeof(T);
    check(cuMemAlloc(&p, bytes));
    check(cuMemcpyHtoD(p, values.data(), bytes));
    return p;
}

static CUdeviceptr cache(std::mt19937& rng, size_t count) {
    std::vector<uint16_t> values(count);
    for (auto& x : values) {
        // Finite signed FP16 values with several exponents and varying mantissas.
        const uint32_t u = rng();
        x = static_cast<uint16_t>((u & 0x8000u) | ((9 + ((u >> 16) % 8)) << 10) | (u & 1023));
    }
    return upload(values);
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    check(cuInit(0));
    CUdevice gpu;
    CUcontext ctx;
    check(cuDeviceGet(&gpu, 0));
    check(cuDevicePrimaryCtxRetain(&ctx, gpu));
    check(cuCtxSetCurrent(ctx));
    CUmodule modules[2];
    CUfunction kernels[2];
    check(cuModuleLoad(&modules[0], argv[1]));
    check(cuModuleLoad(&modules[1], argv[2]));
    check(cuModuleGetFunction(&kernels[0], modules[0], "ink2_gqa_rel_sink_attn_q8_b"));
    check(cuModuleGetFunction(&kernels[1], modules[1], "ink2_gqa_rel_sink_attn_q8_decode"));

    constexpr int max_ctx = 131072;
    std::mt19937 rng(41);
    auto real = [&]() { return static_cast<float>(static_cast<int>(rng() % 2049) - 1024) / 512; };
    constexpr int stride = 8600, max_batch = 8;
    std::vector<float> qrgs(max_batch * stride);
    for (float& x : qrgs) x = real();
    std::vector<uint16_t> rel(16 * 1024);
    for (auto& x : rel) {
        const float f = real() / 8;
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        x = static_cast<uint16_t>(bits >> 16);
    }
    auto q = upload(qrgs), r = upload(rel);
    auto k = cache(rng, size_t(max_ctx) * 1024), v = cache(rng, size_t(max_ctx) * 1024);
    auto pos = upload(std::vector<int>{0});
    constexpr size_t code_bytes = 24 * 128, scale_bytes = 48 * sizeof(float);
    std::array<CUdeviceptr, 2> codes{}, scales{};
    for (int i = 0; i < 2; ++i) {
        check(cuMemAlloc(&codes[i], max_batch * code_bytes));
        check(cuMemAlloc(&scales[i], max_batch * scale_bytes));
    }
    int cases = 0;
    for (int cap : {1024, 1088, 2048, max_ctx}) {
        int global = (cap == max_ctx);
        for (int n : {1, 7, 8, 9, 56, 57, 63, 64, 65, 71, 72, 127, 128, 129,
                      1023, 1024, 1025, 1089, 2049, 8192, 32768, max_ctx}) {
            for (int batch = 1; batch <= max_batch; ++batch) {
                int p = std::min(n - 1, max_ctx - batch);
                int qstride = stride;
                const size_t nc = code_bytes * batch, ns = scale_bytes * batch;
                check(cuMemcpyHtoD(pos, &p, sizeof(p)));
                std::array<std::vector<uint8_t>, 2> output;
                for (int i = 0; i < 2; ++i) {
                    output[i].resize(nc + ns);
                    check(cuMemsetD8(codes[i], i ? 0xaa : 0x55, nc));
                    check(cuMemsetD8(scales[i], i ? 0xaa : 0x55, ns));
                    void* args[]{&q, &qstride, &r, &k, &v, &pos, &cap, &global, &codes[i], &scales[i]};
                    check(cuLaunchKernel(kernels[i], 24, batch, 1, 256, 1, 1, 0, nullptr, args, nullptr));
                    check(cuMemcpyDtoH(output[i].data(), codes[i], nc));
                    check(cuMemcpyDtoH(output[i].data() + nc, scales[i], ns));
                }
                if (output[0] != output[1]) {
                    std::fprintf(stderr, "Different attention output at base=%d cap=%d batch=%d\n", p, cap, batch);
                    return 1;
                }
                ++cases;
            }
        }
    }
    for (auto p : {q, r, k, v, pos, codes[0], codes[1], scales[0], scales[1]}) check(cuMemFree(p));
    for (auto m : modules) check(cuModuleUnload(m));
    check(cuDevicePrimaryCtxRelease(gpu));
    std::printf("%d cases matched (all Q8 codes and FP32 scales)\n", cases);
}
