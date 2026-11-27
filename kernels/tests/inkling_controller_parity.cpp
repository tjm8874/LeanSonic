// Compare both controller stages against a reference cubin, including partial token tiles.
// c++ -O2 -I/usr/local/cuda/include kernels/tests/inkling_controller_parity.cpp -lcuda -o parity
// ./parity reference.cubin candidate.cubin
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
        std::fprintf(stderr, "CUDA error %d: %s\n", result, message ? message : "unknown");
        std::exit(1);
    }
}

template <class T>
static CUdeviceptr upload(const std::vector<T>& data) {
    CUdeviceptr ptr;
    size_t bytes = data.size() * sizeof(T);
    check(cuMemAlloc(&ptr, bytes));
    check(cuMemcpyHtoD(ptr, data.data(), bytes));
    return ptr;
}

static CUfunction kernel(CUmodule mod, const char* name) {
    CUfunction function;
    check(cuModuleGetFunction(&function, mod, name));
    return function;
}

static void compare(CUdeviceptr device, std::vector<float>& ref, bool baseline,
                    const char* stage, int batch, int controllers) {
    std::vector<float> actual(ref.size());
    check(cuMemcpyDtoH(actual.data(), device, actual.size() * sizeof(float)));
    if (baseline) ref = actual;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::memcmp(&ref[i], &actual[i], sizeof(float))) {
            std::fprintf(stderr, "%s B=%d controllers=%d index=%zu: %a vs %a\n",
                         stage, batch, controllers, i, ref[i], actual[i]);
            std::exit(1);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s reference.cubin candidate.cubin\n", argv[0]);
        return 2;
    }
    CUdevice gpu;
    CUcontext context;
    check(cuInit(0));
    check(cuDeviceGet(&gpu, 0));
    check(cuDevicePrimaryCtxRetain(&context, gpu));
    check(cuCtxSetCurrent(context));
    CUmodule ref, candidate;
    check(cuModuleLoad(&ref, argv[1]));
    check(cuModuleLoad(&candidate, argv[2]));
    CUfunction down[] = {kernel(ref, "ink2_ctrl_down_b"), kernel(candidate, "ink2_ctrl_down_b"),
                         kernel(candidate, "ink2_ctrl_down_r8_b"),
                         kernel(candidate, "ink2_ctrl_down_r8_t2_b")};
    CUfunction apply[] = {kernel(ref, "ink2_ctrl_apply_b"), kernel(candidate, "ink2_ctrl_apply_b"),
                          kernel(candidate, "ink2_ctrl_apply_t8_b")};

    std::mt19937 rng(42);
    auto inputs = [&](size_t count) {
        std::vector<float> data(count);
        for (auto& x : data) x = (static_cast<int>(rng() % 10001) - 5000) / 4093.7f;
        return data;
    };
    auto weights = [&](size_t count) {
        std::vector<uint16_t> data(count);
        for (auto& x : data) {
            float w = (static_cast<int>(rng() % 8193) - 4096) / 262144.0f;
            uint32_t bits;
            std::memcpy(&bits, &w, sizeof(w));
            x = static_cast<uint16_t>(bits >> 16);
        }
        return upload(data);
    };
    constexpr int hidden = 2304, rank = 64;
    for (int controllers : {1, 3}) {
        auto wd = weights(controllers * rank * 2 * hidden);
        auto wg = weights(controllers * hidden * rank);
        auto wf = weights(controllers * hidden * rank);
        for (int batch : {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 47, 63, 64, 65, 129, 256}) {
            int rows = controllers * batch;
            auto input = inputs(rows * hidden);
            auto x = upload(input);
            auto msg = upload(inputs(rows * hidden));
            std::vector<float> ref_gate(rows * rank), ref_output(input.size());
            auto gate = upload(ref_gate);
            void* down_args[] = {&x, &msg, &wd, &gate, &batch};
            void* apply_args[] = {&x, &msg, &gate, &wg, &wf, &batch};
            for (int variant = 0; variant < 4; ++variant) {
                const bool wide = variant >= 2;
                const int tile = wide ? 8 : 1;
                const CUfunction apply_kernel = apply[wide ? 2 : variant];
                const int down_blocks = variant == 3 ? controllers * ((batch + 1) / 2) : rows;
                check(cuMemcpyHtoD(x, input.data(), input.size() * sizeof(float)));
                // Poison the output so a skipped row fails even for the reference.
                check(cuMemsetD32(gate, 0x7fc00000, ref_gate.size()));
                check(cuLaunchKernel(down[variant], rank / tile, down_blocks, 1, 256, 1, 1,
                                     0, nullptr, down_args, nullptr));
                compare(gate, ref_gate, variant == 0, "down", batch, controllers);
                check(cuLaunchKernel(apply_kernel, hidden / 256, controllers * ((batch + tile - 1) / tile),
                                     1, 256, 1, 1, 0, nullptr, apply_args, nullptr));
                compare(x, ref_output, variant == 0, "apply", batch, controllers);
            }
            for (auto ptr : {x, msg, gate}) check(cuMemFree(ptr));
        }
        for (auto ptr : {wd, wg, wf}) check(cuMemFree(ptr));
    }
    check(cuModuleUnload(ref));
    check(cuModuleUnload(candidate));
    check(cuDevicePrimaryCtxRelease(gpu));
    std::puts("Both controller stages matched the reference bit-for-bit at every tested shape.");
}
