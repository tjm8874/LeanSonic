// Compare tiled prefill convolution with the serial kernels in the same cubin.
// c++ -O2 -std=c++17 -I/usr/local/cuda/include kernels/tests/inkling_short_conv.cpp -lcuda -o conv-parity
// ./conv-parity candidate.cubin [--quick]
#include <cuda.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

static void check(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* error = nullptr;
        cuGetErrorString(status, &error);
        std::fprintf(stderr, "CUDA: %s\n", error);
        std::exit(1);
    }
}

struct Buffer {
    CUdeviceptr device{};
    std::vector<float> initial;

    Buffer(size_t count, std::mt19937& rng) : initial(count) {
        for (float& x : initial) x = (static_cast<int>(rng() & 0xffffu) - 32768) / 65536.0f;
        check(cuMemAlloc(&device, count * sizeof(float)));
        reset();
    }
    Buffer(const Buffer&) = delete;
    ~Buffer() { cuMemFree(device); }

    void reset() const {
        check(cuMemcpyHtoD(device, initial.data(), initial.size() * sizeof(float)));
    }
    std::vector<uint32_t> read() const {
        std::vector<uint32_t> result(initial.size());
        check(cuMemcpyDtoH(result.data(), device, result.size() * sizeof(uint32_t)));
        return result;
    }
};

static void compare(CUmodule module, int channels, int rows, bool dynamic, bool residual,
                    bool saved_kernel, bool save_output, bool record_sequence) {
    const int input_stride = channels + 16, output_stride = channels + 8;
    const int dyn_stride = (channels + 127) / 128 + 1;
    std::mt19937 rng(27);
    Buffer x(rows * input_stride, rng), weights(4 * channels, rng), dyn(4 * rows * dyn_stride, rng);
    Buffer history(3 * channels, rng), output(rows * output_stride, rng);
    Buffer saved(rows * output_stride, rng), sequence((3 + rows) * channels, rng);
    const Buffer* outputs[] = {&history, &output, &saved, &sequence};
    std::array<std::vector<uint32_t>, 4> expected;
    CUdeviceptr d = dynamic ? dyn.device : 0, res = residual ? output.device : 0;
    CUdeviceptr old = save_output ? saved.device : 0, seq = record_sequence ? sequence.device : 0;
    int ins = input_stride, outs = output_stride, ds = dyn_stride, hd = dynamic, add = residual;
    void* plain[] = {&x.device, &ins, &weights.device, &d, &ds, &history.device, &output.device,
                     &outs, &res, &outs, &seq, &channels, &rows, &hd, &add};
    void* with_save[] = {&x.device, &ins, &weights.device, &d, &ds, &history.device, &output.device,
                         &outs, &res, &outs, &old, &seq, &channels, &rows, &hd, &add};

    for (bool tiled : {false, true}) {
        for (const Buffer* buffer : outputs) buffer->reset();
        const char* name = saved_kernel
            ? (tiled ? "ink2_short_conv4_tiled_save_b" : "ink2_short_conv4_save_b")
            : (tiled ? "ink2_short_conv4_tiled_b" : "ink2_short_conv4_b");
        CUfunction fn;
        check(cuModuleGetFunction(&fn, module, name));
        const int width = tiled ? 32 : 256;
        check(cuLaunchKernel(fn, (channels + width - 1) / width, 1, 1, 256, 1, 1, 0, nullptr,
                             saved_kernel ? with_save : plain, nullptr));
        check(cuCtxSynchronize());
        for (size_t i = 0; i < expected.size(); ++i) {
            auto data = outputs[i]->read(); // includes padding and untouched optional buffers
            if (!tiled) expected[i] = std::move(data);
            else if (data != expected[i]) {
                std::fprintf(stderr, "%s channels=%d rows=%d dynamic=%d residual=%d save=%d seq=%d buffer=%zu\n",
                             name, channels, rows, dynamic, residual, save_output, record_sequence, i);
                std::exit(1);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--quick") != 0)) return 2;
    check(cuInit(0));
    CUdevice device;
    CUcontext ctx;
    check(cuDeviceGet(&device, 0));
    check(cuDevicePrimaryCtxRetain(&ctx, device));
    check(cuCtxSetCurrent(ctx));
    CUmodule module;
    check(cuModuleLoad(&module, argv[1]));

    const bool quick = argc == 3;
    const std::vector<int> widths = quick ? std::vector<int>{129} : std::vector<int>{129, 1024, 2304};
    const std::vector<int> batches = quick ? std::vector<int>{2, 65} : std::vector<int>{1, 2, 3, 7, 63, 64, 65, 129, 512, 1024};
    int count = 0;
    for (int c : widths) for (int b : batches) for (bool seq : {false, true}) {
        compare(module, c, b, false, false, false, false, seq); // SWA K/V
        compare(module, c, b, true, false, false, false, seq);  // global K/V
        compare(module, c, b, false, true, false, false, seq);  // MLP residual
        compare(module, c, b, false, true, true, false, seq);   // optional save
        compare(module, c, b, false, true, true, true, seq);
        count += 5;
    }
    std::printf("%d configurations matched (history, output, saved residual and sequence)\n", count);
    check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(device));
}
