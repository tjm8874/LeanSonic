#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
#define CUDA(call) do {auto err=(call); if(err!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(err));std::exit(2);}} while(0)
constexpr int experts=128, rows=2304, blocks=72;
constexpr size_t count=size_t(experts)*rows*blocks;
template<class T> std::vector<T> load(const char* path,size_t n) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(n*sizeof(T)))std::exit(3);
    f.seekg(0);std::vector<T> v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!f)std::exit(3);return v;
}
template<class T> T* upload(const std::vector<T>& v) {
    T* ptr=nullptr;CUDA(cudaMalloc(&ptr,v.size()*sizeof(T)));CUDA(cudaMemcpy(ptr,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return ptr;
}
__device__ __forceinline__ unsigned expand8(unsigned half) {
    unsigned x=(half|(half<<8))&0x00ff00ffu;
    x=(x|(x<<4))&0x0f0f0f0fu;
    x=(x|(x<<2))&0x33333333u;
    return (x&0x11111111u)*7u | ((x&0x22222222u)<<2);
}
__device__ __forceinline__ unsigned four_signed_bytes(unsigned nibbles) {
    unsigned bytes=0;
    #pragma unroll
    for(int i=0;i<4;++i){
        const unsigned n=(nibbles>>(4*i))&15;
        int q=(n&7) ? ((n&8)?-6:6) : 0;
        bytes |= (unsigned(q)&255u)<<(i*8);
    }
    return bytes;
}
template<bool Packed>
__global__ void project(const void* input,const unsigned char* scales,const int* activations,float* out,int group){
    const int lane=threadIdx.x%32;
    const int row=blockIdx.x*4+threadIdx.x/32;
    const int e=group*8+blockIdx.y;
    const size_t base=(size_t(e)*rows+row)*blocks;
    float sum=0;
    for(int b=lane;b<blocks;b+=32){
        uint2 q;
        if constexpr(Packed){
            const unsigned s=static_cast<const unsigned*>(input)[base+b];
            q=make_uint2(expand8(s&65535u),expand8(s>>16));
        }else q=static_cast<const uint2*>(input)[base+b];
        const int4 a=reinterpret_cast<const int4*>(activations)[blockIdx.y*blocks+b];
        int dot=__dp4a(int(four_signed_bytes(q.x)),a.x,0);
        dot=__dp4a(int(four_signed_bytes(q.x>>16)),a.y,dot);
        dot=__dp4a(int(four_signed_bytes(q.y)),a.z,dot);
        dot=__dp4a(int(four_signed_bytes(q.y>>16)),a.w,dot);
        unsigned s=scales[base+b];
        const float scale=__uint_as_float(((s/8+120)<<23)|((s%8)<<20));
        sum=fmaf(float(dot),scale,sum);
    }
    for(int stride=16;stride>0;stride/=2)sum+=__shfl_down_sync(0xffffffff,sum,stride);
    if(lane==0)out[size_t(e)*rows+row]=sum;
}
int main(){
    auto raw=load<unsigned char>("nibbles.bin",count*8),small=load<unsigned char>("ternary.bin",count*4),sc=load<unsigned char>("scales.bin",count);
    std::vector<signed char> act(8*blocks*16);std::mt19937 rng(73171);for(auto& a:act)a=static_cast<signed char>(int(rng()%241)-120);
    auto a=upload(raw),b=upload(small),scale=upload(sc);auto ac=upload(act);
    float* out;CUDA(cudaMalloc(&out,experts*rows*sizeof(float)));
    cudaEvent_t first,last;CUDA(cudaEventCreate(&first));CUDA(cudaEventCreate(&last));
    auto run=[&](bool packed){
        CUDA(cudaEventRecord(first));
        for(int g=0;g<16;++g) {
            dim3 grid(rows/4,8);
            if(packed)project<true><<<grid,128>>>(b,scale,(int*)ac,out,g);
            else project<false><<<grid,128>>>(a,scale,(int*)ac,out,g);
        }
        CUDA(cudaGetLastError());CUDA(cudaEventRecord(last));CUDA(cudaEventSynchronize(last));
        float ms=0;CUDA(cudaEventElapsedTime(&ms,first,last));return ms;
    };
    std::vector<float> baseline(experts*rows),candidate(baseline.size());
    run(false);CUDA(cudaMemcpy(baseline.data(),out,baseline.size()*4,cudaMemcpyDeviceToHost));
    run(true);CUDA(cudaMemcpy(candidate.data(),out,candidate.size()*4,cudaMemcpyDeviceToHost));
    if(std::memcmp(baseline.data(),candidate.data(),baseline.size()*4))return 4;
    for(int e=0;e<experts;++e)for(int i=0;i<8;++i){
        int row=(i*281+e*37)%rows;double total=0,absolute=0;
        for(int k=0;k<1152;++k){
            size_t n=(size_t(e)*rows+row)*1152+k;
            unsigned v=(raw[n/2]>>(4*(n%2)))&15;
            double w=v==0?0:(v==7?6:-6);
            unsigned s=sc[n/16];
            double term=w*std::ldexp(1.+double(s%8)/8.,int(s/8)-7)*act[(e%8)*1152+k];
            total+=term;absolute+=std::abs(term);
        }
        if(std::abs(candidate[e*rows+row]-total)>std::max(1e-7,absolute*1e-6))return 5;
    }
    for(int cycle=0;cycle<9;++cycle)for(int leg=0;leg<4;++leg){
        bool packed=(leg==1||leg==2)^(cycle%2!=0);
        std::printf("%d,%d,%s,%.6f\n",cycle,leg,packed?"two":"four",run(packed));
    }
    CUDA(cudaEventDestroy(first));CUDA(cudaEventDestroy(last));CUDA(cudaFree(out));CUDA(cudaFree(a));CUDA(cudaFree(b));CUDA(cudaFree(scale));CUDA(cudaFree(ac));return 0;
}
