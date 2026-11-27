#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include "pair.cuh"
#define CHECK(call) do { auto err=(call); if(err!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(err));return 2;} } while(0)
__global__ void expand(const unsigned* src, uint2* out, unsigned pairs) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<pairs)out[i]=nvfp4_scale_pair(src,2*i);
}
int main(){
    std::mt19937 rng(3973);
    for(unsigned pairs: {1u,2u,3u,9u,18u,129u,2310u,65537u}) {
        const unsigned codes=pairs*8u;
        std::vector<unsigned> packed((codes*6u+31)/32,0);
        std::vector<unsigned char> original(codes),decoded(codes);
        for(unsigned i=0;i<codes;++i){
            unsigned x=i%9==0?63u:i%7==0?0u:rng()%64;original[i]=64+x;
            for(unsigned j=0;j<6;++j)packed[(i*6+j)/32]|=((x>>j)&1u)<<((i*6+j)%32);
        }
        unsigned* src=nullptr;uint2* out=nullptr;
        CHECK(cudaMalloc(&src,packed.size()*sizeof(unsigned)));CHECK(cudaMalloc(&out,pairs*sizeof(uint2)));
        CHECK(cudaMemcpy(src,packed.data(),packed.size()*4,cudaMemcpyHostToDevice));
        expand<<<(pairs+127)/128,128>>>(src,out,pairs);CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(decoded.data(),out,codes,cudaMemcpyDeviceToHost));
        if(decoded!=original)return 1;
        CHECK(cudaFree(src));CHECK(cudaFree(out));
    }
    std::puts("All packed scales match the CPU, including both alignments and exact allocation ends.");
}
