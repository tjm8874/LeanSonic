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
    std::vector<int> counts(experts,0), pairs(6*64*experts,-1);
    const int sizes[]={1,63,64,65,127,128,129,159,160,161,319};
    int active=0;
    for(int i=0;i<11;++i) {
        int e=(i*37+5)%128;
        counts[e]=sizes[i];
        for(int j=0;j<sizes[i];++j) pairs[((j/64)*128+e)*64+j%64]=active++;
    }
    // Dense sorted activations, deliberately reverse the scatter destination.
    for(int& id:pairs) if(id>=0) id=active-1-id;
    std::mt19937 rng(20261127);
    std::vector<unsigned char> activations(active*1152), xs(active*144);
    std::vector<float> gs(active);
    for(auto& v:activations)v=rng()%256;
    for(auto& v:xs)v=48+rng()%18;
    for(auto& v:gs)v=(4+rng()%16)/16.f;
    RT(cudaSetDevice(0));RT(cudaFree(nullptr));
    CUmodule mod;DR(cuModuleLoad(&mod,"engine.cubin"));
    auto dn=upload(raw), dt=upload(ternary);auto ds=upload(sc6);auto da=upload(amplitude);
    auto dx=upload(activations), dsc=upload(xs);auto dg=upload(gs);
    auto dc=upload(counts), dids=upload(pairs);
    size_t size=(active+2)*rows;
    std::vector<unsigned short> ref(size),candidate(size);
    unsigned short* allocation=nullptr;RT(cudaMalloc(&allocation,size*2));
    auto out=allocation+rows;
    for(int tile:{128,160}){
        int grids=0;
        for(auto count:counts)grids+=(count+tile-1)/tile;
        for(bool compact:{false,true}){
            CUfunction fn;std::string name=compact?"ink2_moe_ternary_tc_down":"ink2_moe_nvfp4_tc_down";
            if(tile==160)name+="_t160";
            name+="_packed_scales";
            DR(cuModuleGetFunction(&fn,mod,name.c_str()));
            const unsigned smem=tile==160?57616:62480;
            DR(cuFuncSetAttribute(fn,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,smem));
            RT(cudaMemset(allocation,0xff,size*2));
            int begin=0,end=128;auto weights=compact?dt:dn;
            void* args[]={&weights,&ds,&da,&dx,&dsc,&dg,&dc,&dids,&begin,&end,&out};
            DR(cuLaunchKernel(fn,tile==160?12:9,grids+2,1,tile==160?384:512,1,1,smem,nullptr,args,nullptr));
            auto& dest=compact?candidate:ref;
            RT(cudaMemcpy(dest.data(),allocation,size*2,cudaMemcpyDeviceToHost));
            size_t nonzero=0;
            for(size_t i=0;i<size;++i){
                if(i<rows || i>=size-rows){if(dest[i]!=0xffffu)return 4;}
                else {
                    if((dest[i]&0x7c00u)==0x7c00u)return 5;
                    nonzero+=bool(dest[i]&0x7fffu);
                }
            }
            if(nonzero<rows*active/2)return 6;
        }
        if(ref!=candidate)return 7;
        std::printf("tile=%d active=%d exact=%zu guards=%zu\n",tile,active,rows*active,2*rows);
    }
    DR(cuModuleUnload(mod));
    for(void* p:std::vector<void*>{allocation,dn,dt,ds,da,dx,dsc,dg,dc,dids})RT(cudaFree(p));
}
