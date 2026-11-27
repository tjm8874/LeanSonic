// Build with the CUDA driver SDK. Test the actual split and combine symbols
// in the supplied cubin against uniform attention with an analytic solution.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <vector>
void check(CUresult v) {
    if (v != CUDA_SUCCESS) {
        const char *s;
        cuGetErrorString(v, &s);
        std::fprintf(stderr, "%s\n", s);
        std::exit(2);
    }
}
struct Buffer {
    CUdeviceptr p;
    explicit Buffer(size_t n) {
        check(cuMemAlloc(&p, n));
        check(cuMemsetD8(p, 0, n));
    }
    ~Buffer() { cuMemFree(p); }
    Buffer(const Buffer &) = delete;
};
int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s /path/to/inkling2_sm120.cubin\n", argv[0]);
        return 2;
    }
    check(cuInit(0));
    CUdevice device;
    CUcontext ctx;
    check(cuDeviceGet(&device, 0));
    check(cuDevicePrimaryCtxRetain(&ctx, device));
    check(cuCtxSetCurrent(ctx));
    CUmodule mod;
    check(cuModuleLoad(&mod, argv[1]));
    CUfunction combine;
    check(cuModuleGetFunction(&combine, mod, "ink2_flash_attn_split_combine_q8_b"));
    constexpr int batch = 3, heads = 24, dim = 128;
    std::puts("kv,length,value,nonfinite_partials,nonfinite_outputs,max_error_over_finite_outputs,status");
    bool all_passed = true;
    for (bool int8_kv : {false, true}) {
        CUfunction split;
        check(cuModuleGetFunction(&split, mod,
                                  int8_kv ? "ink2_flash_attn_tc_split_b_i8" : "ink2_flash_attn_tc_split_b"));
        check(cuFuncSetAttribute(split, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 64512));
        for (int length : {4096, 98304, 131072}) {
            int cap = length, pos = length - batch, b = batch, global = 1, stride = 8600, splits = 6;
            size_t elements = size_t(cap) * 8 * dim;
            Buffer q(b * stride * 4), projection(16 * 1024 * 2), k(elements * (int8_kv ? 1 : 2)),
                v(elements * (int8_kv ? 1 : 2));
            Buffer key_scales(size_t(cap) * 8 * 4), value_scales(size_t(cap) * 8 * 4), position(4);
            // Alternate the sign by test length, preserving a closed-form value per head.
            int value = length == 131072 ? -8 : 8;
            if (int8_kv)
                check(cuMemsetD8(v.p, static_cast<uint8_t>(value), elements));
            else
                check(cuMemsetD16(v.p, value < 0 ? 0xc800 : 0x4800, elements));
            check(cuMemsetD32(key_scales.p, 0x3f800000, cap * 8));
            check(cuMemsetD32(value_scales.p, 0x3f800000, cap * 8));
            check(cuMemcpyHtoD(position.p, &pos, 4));
            size_t parts = size_t(splits) * b * heads;
            Buffer partial(parts * dim * 2), ml(parts * 8), codes(b * heads * dim), scales(b * heads * 2 * 4);
            void *fp16_args[] = {&q.p,    &stride, &projection.p, &k.p, &v.p, &position.p, &cap,
                                 &global, &b,      &partial.p,    &ml.p};
            void *int8_args[] = {
                &q.p,        &stride, &projection.p, &k.p, &v.p,       &key_scales.p, &value_scales.p,
                &position.p, &cap,    &global,       &b,   &partial.p, &ml.p};
            check(cuLaunchKernel(split, 8, 1, splits, 384, 1, 1, int8_kv ? 63872 : 64384, nullptr,
                                 int8_kv ? int8_args : fp16_args, nullptr));
            void *combine_args[] = {&partial.p, &ml.p, &q.p, &stride, &b, &splits, &codes.p, &scales.p};
            check(cuLaunchKernel(combine, heads, b, 1, 64, 1, 1, 0, nullptr, combine_args, nullptr));
            std::vector<uint16_t> partial_bits(parts * dim);
            std::vector<int8_t> quantized(b * heads * dim);
            std::vector<float> scale(b * heads * 2);
            check(cuMemcpyDtoH(partial_bits.data(), partial.p, partial_bits.size() * 2));
            check(cuMemcpyDtoH(quantized.data(), codes.p, quantized.size()));
            check(cuMemcpyDtoH(scale.data(), scales.p, scale.size() * 4));
            int nonfinite_partials = 0, nonfinite_outputs = 0;
            double error = 0;
            for (auto t : partial_bits)
                nonfinite_partials += ((t & 0x7c00u) == 0x7c00u);
            for (size_t i = 0; i < quantized.size(); ++i) {
                int row = int(i / (heads * dim)), keys = pos + row + 1;
                // Q, K, relative logits, sink and output gates are zero;
                // keys and the sink have equal softmax mass, gate sigmoid is 1/2.
                double expected = 0.5 * value * keys / (keys + 1.);
                double actual = double(quantized[i]) * scale[i / 64];
                if (!std::isfinite(actual))
                    ++nonfinite_outputs;
                else
                    error = std::max(error, std::abs(actual - expected));
            }
            bool ok = nonfinite_partials == 0 && nonfinite_outputs == 0 && error < 0.005;
            std::printf("%s,%d,%d,%d,%d,%.8f,%s\n", int8_kv ? "int8" : "fp16", length, value,
                        nonfinite_partials, nonfinite_outputs, error, ok ? "PASS" : "FAIL");
            all_passed &= ok;
        }
    }
    return all_passed ? 0 : 1;
}
