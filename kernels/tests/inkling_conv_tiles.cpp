// c++ -O2 -I/usr/local/cuda/include kernels/tests/inkling_conv_tiles.cpp -lcuda -o conv-tiles
// ./conv-tiles baseline.cubin candidate.cubin
#include <cuda.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "CUDA error %d: %s\n", result, message ? message : "unknown");
    std::exit(1);
}

struct Buffer {
    explicit Buffer(const std::vector<float>& initial) : bytes(initial.size() * sizeof(float)) {
        check(cuMemAlloc(&ptr, bytes));
        check(cuMemcpyHtoD(ptr, initial.data(), bytes));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { check(cuMemFree(ptr)); }
    std::vector<float> read() const {
        std::vector<float> values(bytes / sizeof(float));
        check(cuMemcpyDtoH(values.data(), ptr, bytes));
        return values;
    }
    CUdeviceptr ptr;
    size_t bytes;
};

static void parity(const CUfunction (&kernels)[2], int channels, int batch, bool dynamic, bool residual,
                   bool save, bool record, std::mt19937& rng) {
    auto values = [&](size_t count) {
        std::vector<float> data(count);
        for (auto& v : data) v = (static_cast<int>(rng() % 8193) - 4096) / 4096.0f;
        return data;
    };
    int input_stride = channels + 9, output_stride = channels + 5;
    int dyn_stride = (channels + 127) / 128 + 1;
    Buffer input(values(batch * input_stride)), weights(values(4 * channels));
    Buffer dynamics(values(batch * dyn_stride * 4));
    auto initial = values(batch * output_stride), hist = values(3 * channels);
    std::vector<float> reference[4];
    for (int i = 0; i < 2; ++i) {
        Buffer output(initial), history(hist);
        Buffer saved(std::vector<float>(batch * output_stride, -12345.0f));
        Buffer seq(std::vector<float>((3 + batch) * channels, -12345.0f));
        CUdeviceptr d = dynamic ? dynamics.ptr : 0, r = residual ? output.ptr : 0;
        CUdeviceptr s = save ? saved.ptr : 0, q = record ? seq.ptr : 0;
        int hd = dynamic, dr = residual;
        void* args[] = {&input.ptr, &input_stride, &weights.ptr, &d, &dyn_stride, &history.ptr,
                        &output.ptr, &output_stride, &r, &output_stride, &s, &q,
                        &channels, &batch, &hd, &dr};
        const unsigned gx = i ? (channels + 31) / 32 : (channels + 255) / 256;
        const unsigned gy = i ? (batch + 63) / 64 : 1;
        check(cuLaunchKernel(kernels[i], gx, gy, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        const Buffer* buffers[] = {&output, &history, &saved, &seq};
        for (int j = 0; j < 4; ++j) {
            auto data = buffers[j]->read();
            if (!i) reference[j] = std::move(data);
            else if (std::memcmp(reference[j].data(), data.data(), buffers[j]->bytes) != 0) {
                std::fprintf(stderr, "conv mismatch: B=%d channels=%d buffer=%d dynamic=%d residual=%d save=%d record=%d\n",
                             batch, channels, j, dynamic, residual, save, record);
                std::exit(1);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    CUdevice device;
    CUcontext context;
    check(cuInit(0));
    check(cuDeviceGet(&device, 0));
    check(cuDevicePrimaryCtxRetain(&context, device));
    check(cuCtxSetCurrent(context));
    CUmodule mods[2];
    CUfunction kernels[2];
    for (int i = 0; i < 2; ++i) {
        check(cuModuleLoad(&mods[i], argv[i + 1]));
        check(cuModuleGetFunction(&kernels[i], mods[i], i ? "ink2_short_conv4_wide_b" : "ink2_short_conv4_save_b"));
    }
    std::mt19937 rng(42);
    for (int b : {64, 65, 71, 72, 127, 128, 129, 255, 257, 1024}) {
        for (int flags = 0; flags < 16; ++flags) {
            parity(kernels, 137, b, flags & 1, flags & 2, flags & 4, flags & 8, rng);
        }
    }
    parity(kernels, 1024, 1024, true, false, false, false, rng);
    parity(kernels, 2304, 1024, false, true, true, false, rng);
    for (auto m : mods) check(cuModuleUnload(m));
    check(cuDevicePrimaryCtxRelease(device));
    std::puts("162 cases matched bitwise for output, row padding, final history, saved residual and rewind sequence");
}
