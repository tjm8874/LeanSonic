#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
#define RT(call) do {auto e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));std::exit(2);}} while(0)
#define DR(call) do {auto e=(call); if(e!=CUDA_SUCCESS){const char* msg=nullptr;cuGetErrorString(e,&msg);std::fprintf(stderr,"%s: %s\n",#call,msg);std::exit(2);}} while(0)
constexpr size_t experts=128, rows=2304, blocks=72;
template<class T> std::vector<T> read(const char* name,size_t count) {
    std::ifstream f(name,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(count*sizeof(T)))std::exit(3);
    std::vector<T> values(count);f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(values.data()),count*sizeof(T)))std::exit(3);
    return values;
}
template<class T> T* upload(const std::vector<T>& values) {
    T* d=nullptr;RT(cudaMalloc(&d,values.size()*sizeof(T)));
    RT(cudaMemcpy(d,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));return d;
}
int main(){
    auto n=experts*rows*blocks;
    auto raw=read<unsigned char>("nibbles.bin",n*8);
    auto ternary=read<unsigned char>("ternary.bin",n*4);
    auto scale=read<unsigned char>("scales.bin",n);
    auto amplitude=read<float>("amax.bin",experts);
    std::vector<unsigned> sc6((n*6+31)/32);
    for(size_t i=0;i<n;++i){
        if(scale[i]<64||scale[i]>126)return 3;
        const unsigned v=scale[i]-64,bit=(i*6)%32;
        size_t word=(i*6)/32;sc6[word]|=v<<bit;
        if(bit>26)sc6[word+1]|=v>>(32-bit);
    }
    std::vector<signed char> act(8*1152);
    std::vector<float> xs(8*blocks),gate(8);
    std::vector<int> selected(128);
    std::mt19937 rng(381771);
    for(auto& v:act)v=int(rng()%241)-120;
    for(auto& v:xs)v=float(1+rng()%64)/32.f;
    for(int k=0;k<8;++k)gate[k]=float(k+1)/36.f;
    for(int k=0;k<128;++k)selected[k]=(k*37+5)%128;
    RT(cudaSetDevice(0));RT(cudaFree(nullptr));
    CUmodule mod;DR(cuModuleLoad(&mod,"candidate.cubin"));
    CUfunction orig,packed;
    DR(cuModuleGetFunction(&orig,mod,"ink2_moe_nvfp4_dp4a_down_xw_packed_scales"));
    DR(cuModuleGetFunction(&packed,mod,"ink2_moe_ternary_dp4a_down_xw_packed_scales"));
    auto dp=upload(raw),dt=upload(ternary);auto ds=upload(sc6);auto da=upload(amplitude);
    auto dx=upload(act);auto dscale=upload(xs);auto dids=upload(selected);auto dg=upload(gate);
    std::vector<float> baseline(16*rows),candidate(baseline.size());
    float* result=nullptr;RT(cudaMalloc(&result,baseline.size()*sizeof(float)));
    cudaEvent_t start,end;RT(cudaEventCreate(&start));RT(cudaEventCreate(&end));
    auto run=[&](bool use_small){
        RT(cudaEventRecord(start));
        for(int group=0;group<16;++group){
            auto weights=use_small?dt:dp;auto ids=dids+8*group;auto out=result+group*rows;
            void* args[]={&weights,&ds,&da,&dx,&dscale,&ids,&dg,&out};
            DR(cuLaunchKernel(use_small?packed:orig,1152,1,1,256,1,1,0,nullptr,args,nullptr));
        }
        RT(cudaEventRecord(end));RT(cudaEventSynchronize(end));float ms=0;
        RT(cudaEventElapsedTime(&ms,start,end));return ms;
    };
    run(false);RT(cudaMemcpy(baseline.data(),result,baseline.size()*4,cudaMemcpyDeviceToHost));
    run(true);RT(cudaMemcpy(candidate.data(),result,candidate.size()*4,cudaMemcpyDeviceToHost));
    if(std::memcmp(baseline.data(),candidate.data(),baseline.size()*4))return 4;
    for(float x:candidate)if(!std::isfinite(x))return 4;
    for(int group=0;group<16;++group)for(int i=0;i<16;++i){
        int row=(group*37+i*139)%rows;double sum=0,bound=0;
        for(int k=0;k<8;++k){
            int ex=selected[8*group+k];
            for(int j=0;j<1152;++j){
                size_t index=(ex*rows+row)*1152+j;
                auto v=(raw[index/2]>>(4*(index%2)))&15u;
                if(v!=0 && v!=7 && v!=15)return 5;
                unsigned s=scale[index/16];
                double unit=std::ldexp(1.+double(s%8)/8.,int(s/8)-7);
                double term=(v==7?6:v==15?-6:0)*unit*act[k*1152+j]*xs[k*blocks+j/16]*amplitude[ex]*gate[k]/2688.;
                sum+=term;bound+=std::abs(term);
            }
        }
        if(std::abs(candidate[group*rows+row]-sum)>std::max(1.e-9,bound*1.e-6))return 5;
    }
    for(int c=0;c<9;++c)for(int leg=0;leg<4;++leg){
        bool use_small=(leg==1||leg==2)^(c%2!=0);
        std::printf("%d,%d,%s,%.6f\n",c,leg,use_small?"two":"four",run(use_small));
    }
    RT(cudaEventDestroy(start));RT(cudaEventDestroy(end));DR(cuModuleUnload(mod));
    for(void* p:std::vector<void*>{result,dp,dt,ds,da,dx,dscale,dids,dg})RT(cudaFree(p));
}
