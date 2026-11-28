// Exercise the shipped cubin with exactly sized buffers. Build with -lcuda.
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "CUDA error %d (%s)\n", int(result), message ? message : "unknown");
    std::exit(1);
}

static float as_float(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

static float bf16_round(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof word);
    return as_float((word + 0x7fff + ((word >> 16) & 1)) >> 16);
}

template<class T> static CUdeviceptr upload(const std::vector<T>& data) {
    CUdeviceptr p;
    check(cuMemAlloc(&p, data.size() * sizeof(T)));
    check(cuMemcpyHtoD(p, data.data(), data.size() * sizeof(T)));
    return p;
}

static void exercise(CUfunction fn, int rows, int cols, int batch, bool red) {
    std::vector<int8_t> weights(size_t(rows) * cols);
    std::vector<uint16_t> scales(weights.size() / 32);
    std::vector<float> inputs(size_t(batch) * cols);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = int((i * 61 + i / 13) % 255) - 127;
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = i % 17 ? 0x3b80 + (i * 37 % 512) : 0;
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = (int((i * 17 + i / 37) % 37) - 18) / 16.0f;
    std::vector<float> outputs(size_t(batch) * rows - (red ? 1 : 0), NAN);
    auto w = upload(weights), s = upload(scales), x = upload(inputs), y = upload(outputs);
    void* args[] = {&w, &s, &x, &y, &rows, &cols};
    check(cuLaunchKernel(fn, (rows + 7) / 8, batch, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    CUresult result = cuCtxSynchronize();
    if (red) return; // memcheck owns the one-float-short fault and its exit code.
    check(result);
    check(cuMemcpyDtoH(outputs.data(), y, outputs.size() * sizeof(float)));
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            double expected = 0;
            for (int c = 0; c < cols; ++c) {
                size_t i = size_t(r) * cols + c;
                expected += double(bf16_round(weights[i] * as_float(scales[i / 32]))) * inputs[size_t(b) * cols + c];
            }
            float actual = outputs[size_t(b) * rows + r];
            double difference = std::abs(actual - expected);
            if (!std::isfinite(actual) || difference > 1e-3) {
                std::fprintf(stderr, "row %d batch %d: actual %.9g reference %.9g\n", r, b, actual, expected);
                std::exit(1);
            }
            worst = std::fmax(worst, difference);
        }
    }
    std::printf("rows=%d cols=%d batch=%d max_abs_err=%.9g\n", rows, cols, batch, worst);
    for (auto ptr : {w, s, x, y}) check(cuMemFree(ptr));
}

int main(int argc, char** argv) {
    if (argc != 3 || (std::strcmp(argv[2], "green") && std::strcmp(argv[2], "red"))) return 2;
    check(cuInit(0));
    CUdevice device;
    check(cuDeviceGet(&device, 0));
    CUcontext context;
    check(cuDevicePrimaryCtxRetain(&context, device));
    check(cuCtxSetCurrent(context));
    CUmodule module;
    check(cuModuleLoad(&module, argv[1]));
    CUfunction fn;
    check(cuModuleGetFunction(&fn, module, "q38_dense_q8_gemv"));
    if (!std::strcmp(argv[2], "red")) {
        exercise(fn, 9, 160, 2, true);
        return 0;
    }
    for (int batch : {1, 2, 3, 4, 8, 32, 64}) {
        for (int cols : {32, 160, 2560, 10240}) exercise(fn, 9, cols, batch, false);
    }
    check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(device));
}
