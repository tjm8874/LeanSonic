// Host-side check of the BF16 projection probe, including batched strides and its V offset.
#include <cuda.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
}

template <class T> CUdeviceptr upload(const std::vector<T>& values) {
    CUdeviceptr ptr;
    size_t bytes = sizeof(T) * values.size();
    check(cuMemAlloc(&ptr, bytes));
    check(cuMemcpyHtoD(ptr, values.data(), bytes));
    return ptr;
}

void run(CUfunction kernel, int rows, int cols, int batch, int offset, int stride) {
    std::vector<uint16_t> w(rows * cols);
    std::vector<float> x(batch * cols);
    std::vector<float> expected((batch - 1) * stride + offset + rows, -99999.f);
    for (int i = 0; i < rows * cols; ++i) {
        float value = float((i * 11 + i / cols) % 7 - 3);
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        w[i] = bits >> 16;
    }
    for (int i = 0; i < batch * cols; ++i) x[i] = float((i * 7 + i / cols) % 17 - 8) * 0.25f;
    std::vector<float> got(expected);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            float dot = 0;
            for (int c = 0; c < cols; ++c) {
                uint32_t bits = uint32_t(w[r * cols + c]) << 16;
                float value;
                std::memcpy(&value, &bits, 4);
                dot += x[b * cols + c] * value;
            }
            expected[b * stride + offset + r] = dot;
        }
    }
    CUdeviceptr weights = upload(w), input = upload(x), storage = upload(got);
    CUdeviceptr out = storage + offset * sizeof(float);
    void* args[] = {&weights, &input, &out, &rows, &cols, &stride};
    check(cuLaunchKernel(kernel, (rows+7)/8, batch, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    check(cuCtxSynchronize());
    check(cuMemcpyDtoH(got.data(), storage, got.size()*sizeof(float)));
    if (got != expected) throw std::runtime_error("projection or untouched output padding mismatch");
    for (auto ptr : {weights,input,storage}) check(cuMemFree(ptr));
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    check(cuInit(0));
    CUdevice device;
    check(cuDeviceGet(&device, 0));
    CUcontext ctx;
    check(cuDevicePrimaryCtxRetain(&ctx,device));
    check(cuCtxSetCurrent(ctx));
    CUmodule mod;
    check(cuModuleLoad(&mod, argv[1]));
    CUfunction kernel;
    check(cuModuleGetFunction(&kernel,mod,"q38_probe_bf16_projection"));
    run(kernel, 512, 2560, 1, 12800, 13952);
    run(kernel, 512, 2560, 3, 12800, 13952);
    run(kernel, 2560, 6144, 3, 0, 2560);
    run(kernel, 13, 2560, 5, 3, 23);
    check(cuModuleUnload(mod));
    check(cuDevicePrimaryCtxRelease(device));
    std::cout << "Four projection configurations matched the CPU reference and guard values.\n";
}
