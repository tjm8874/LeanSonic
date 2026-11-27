#include <cstdio>
#include <vector>
#include "inkling2_sm120.cu"

static void require_cuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}

__global__ void test_indices(const uint4* table, unsigned* outputs) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    outputs[i] = expand_four_indices(table[0], i);
}

int main() {
    constexpr unsigned words = 319;
    constexpr unsigned experts = 128;
    constexpr unsigned selected[] = {124, 81, 127, 0, 11, 42, 64, 97};
    const unsigned char table[16] = {0x00,0x80,0x01,0x81,0x07,0x87,0x08,0x88,0x7e,0xfe,0x34,0xb4,0x29,0xa9,0x3f,0xbf};
    std::vector<unsigned char> plane(16 + experts * words * 4);
    std::copy(table, table + 16, plane.begin());
    for (size_t i = 16; i < plane.size(); ++i) plane[i] = static_cast<unsigned char>(i*13 + i/256);
    unsigned char *source, *expanded;
    unsigned *outputs, *ids;
    require_cuda(cudaMalloc(&source, plane.size()));
    require_cuda(cudaMalloc(&expanded, experts * words * 8));
    require_cuda(cudaMalloc(&outputs, 65536 * sizeof(unsigned)));
    require_cuda(cudaMalloc(&ids, sizeof(selected)));
    require_cuda(cudaMemcpy(source, plane.data(), plane.size(), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(ids, selected, sizeof(selected), cudaMemcpyHostToDevice));
    test_indices<<<256,256>>>(reinterpret_cast<uint4*>(source), outputs);
    require_cuda(cudaGetLastError());
    std::vector<unsigned> values(65536);
    require_cuda(cudaMemcpy(values.data(),outputs,values.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for (unsigned i=0; i<values.size(); ++i) {
        unsigned expected=0;
        for (unsigned j=0;j<4;++j) expected |= unsigned(table[(i>>(4*j))&15]) << (8*j);
        if (expected!=values[i]) return 3;
    }
    std::vector<unsigned char> actual(experts * words * 8);
    for (bool subset : {false, true}) {
        require_cuda(cudaMemset(expanded, 0xcd, actual.size()));
        ink2_expand_mxfp8_indices<<<dim3(3,subset?8:experts),256>>>(
            source,reinterpret_cast<ushort4*>(expanded),subset?reinterpret_cast<int*>(ids):nullptr,words);
        require_cuda(cudaGetLastError());
        require_cuda(cudaMemcpy(actual.data(),expanded,actual.size(),cudaMemcpyDeviceToHost));
        for (size_t i=0;i<actual.size();++i) {
            unsigned expert=i/(words*8);
            bool present=!subset;
            for (auto id:selected) present |= expert==id;
            unsigned char packed=plane[16+i/2];
            unsigned char expected=present?table[(packed>>(4*(i%2)))&15]:0xcd;
            if (actual[i]!=expected) return 4;
        }
    }
    require_cuda(cudaFree(source)); require_cuda(cudaFree(expanded)); require_cuda(cudaFree(outputs)); require_cuda(cudaFree(ids));
    std::puts("PASS 65536 selectors; 128 contiguous experts; 8 selected experts; untouched sentinels");
}
