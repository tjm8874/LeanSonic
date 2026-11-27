#include "inkling2_sm120.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checked(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}

__global__ void compare_scale_pairs(const unsigned* raw, const unsigned* packed, int n, unsigned* bad) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(n)) return;
    const unsigned ref = (raw[i >> 1] >> ((i & 1) * 16)) & 0xffff;
    const unsigned prior = (expert_scale_word<true, 6, 64>(packed, i >> 1) >> ((i & 1) * 16)) & 0xffff;
    const unsigned fields = nvfp4_gate_scale_input<true>(packed, i);
    const unsigned word = nvfp4_gate_scale_input<false>(raw, i);
    const float expected0 = e4m3_byte_to_f32(ref & 255) * 0.5f;
    const float expected1 = e4m3_byte_to_f32(ref >> 8) * 0.5f;
    if (prior != ref || word != raw[i >> 1] ||
        __float_as_uint(nvfp4_packed_half_scale(fields)) != __float_as_uint(expected0) ||
        __float_as_uint(nvfp4_packed_half_scale(fields >> 6)) != __float_as_uint(expected1))
        atomicAdd(bad, 1u);
}

int main() {
    constexpr unsigned n = 4096 * 8;
    std::vector<unsigned> raw(n / 2), packed(n * 12 / 32 + 1);
    for (unsigned i = 0; i < n; ++i) {
        // The odd multiplier permutes all 4096 pairs separately at each alignment.
        unsigned v = ((i / 8) * 997 + (i % 8) * 71) & 4095;
        unsigned bytes = (64 + (v & 63)) | ((64 + (v >> 6)) << 8);
        raw[i >> 1] |= bytes << ((i & 1) * 16);
        unsigned bit = i * 12, shift = bit % 32;
        packed[bit / 32] |= v << shift;
        if (shift > 20) packed[bit / 32 + 1] |= v >> (32 - shift);
    }
    unsigned *d_raw, *d_packed, *d_bad, bad;
    checked(cudaMalloc(&d_raw, raw.size() * sizeof(unsigned)));
    checked(cudaMalloc(&d_packed, packed.size() * sizeof(unsigned)));
    checked(cudaMalloc(&d_bad, sizeof(bad)));
    checked(cudaMemcpy(d_raw, raw.data(), raw.size()*sizeof(unsigned), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(d_packed, packed.data(), packed.size()*sizeof(unsigned), cudaMemcpyHostToDevice));
    checked(cudaMemset(d_bad, 0, sizeof(bad)));
    compare_scale_pairs<<<n/256,256>>>(d_raw, d_packed, n, d_bad);
    checked(cudaGetLastError());
    checked(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
    checked(cudaFree(d_bad)); checked(cudaFree(d_packed)); checked(cudaFree(d_raw));
    std::printf("pairs=%u mismatches=%u\n", n, bad);
    return bad != 0;
}
