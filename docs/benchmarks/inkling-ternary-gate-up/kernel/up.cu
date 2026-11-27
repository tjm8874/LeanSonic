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
constexpr size_t experts=128, rows=2304, blocks=144;
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
    std::vector<unsigned> sc6((n*6+31)/32+1);
    for(size_t i=0;i<n;++i){
        if(scale[i]<64||scale[i]>126)return 3;
        const unsigned v=scale[i]-64,bit=(i*6)%32;
        size_t word=(i*6)/32;sc6[word]|=v<<bit;
        if(bit>26)sc6[word+1]|=v>>(32-bit);
    }
    std::vector<signed char> act(2304);
    std::vector<float> xs(144);
    std::vector<int> selected(128);
    std::mt19937 rng(381771);
    for (auto& v:act) v=int(rng()%241)-120;
    for (auto& v:xs) v=float(1+rng()%64)/32.f;
    for (int k=0;k<128;++k) selected[k]=(k*37+5)%128;
    RT(cudaSetDevice(0)); RT(cudaFree(nullptr));
    CUmodule mod; DR(cuModuleLoad(&mod,"candidate.cubin"));
    CUfunction orig,packed;
    DR(cuModuleGetFunction(&orig,mod,"ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales"));
    DR(cuModuleGetFunction(&packed,mod,"ink2_moe_ternary_dp4a_up_half_q8_packed_scales"));
    auto dp=upload(raw),dt=upload(ternary); auto ds=upload(sc6); auto da=upload(amplitude);
    auto dx=upload(act); auto dscale=upload(xs); auto dids=upload(selected);
    constexpr size_t tokens=16, channels=8*1152, groups=channels/16;
    std::vector<signed char> base_q(tokens*channels), new_q(base_q.size());
    std::vector<float> base_s(tokens*groups), new_s(base_s.size());
    signed char* dq=nullptr; float* dsout=nullptr;
    RT(cudaMalloc(&dq,base_q.size())); RT(cudaMalloc(&dsout,base_s.size()*sizeof(float)));
    cudaEvent_t begin,end; RT(cudaEventCreate(&begin)); RT(cudaEventCreate(&end));
    auto run=[&](bool small){
        RT(cudaEventRecord(begin));
        for(int g=0;g<16;++g){
            auto w=small?dt:dp; auto ids=dids+8*g;
            auto oq=dq+g*channels; auto os=dsout+g*groups;
            void* args[]={&w,&ds,&da,&dx,&dscale,&ids,&oq,&os};
            DR(cuLaunchKernel(small?packed:orig,72,8,1,512,1,1,0,nullptr,args,nullptr));
        }
        RT(cudaEventRecord(end)); RT(cudaEventSynchronize(end)); float elapsed=0;
        RT(cudaEventElapsedTime(&elapsed,begin,end)); return elapsed;
    };
    auto copy=[&](auto& q,auto& s){
        RT(cudaMemcpy(q.data(),dq,q.size(),cudaMemcpyDeviceToHost));
        RT(cudaMemcpy(s.data(),dsout,s.size()*sizeof(float),cudaMemcpyDeviceToHost));
    };
    RT(cudaMemset(dq,0xa5,base_q.size())); RT(cudaMemset(dsout,0xa5,base_s.size()*sizeof(float)));
    run(false); copy(base_q,base_s);
    RT(cudaMemset(dq,0x5a,base_q.size())); RT(cudaMemset(dsout,0x5a,base_s.size()*sizeof(float)));
    run(true); copy(new_q,new_s);
    if(base_q!=new_q||std::memcmp(base_s.data(),new_s.data(),new_s.size()*sizeof(float)))return 4;
    for(float x:new_s)if(!std::isfinite(x)||x<=0)return 4;
    // Re-evaluate 512 gate/up dots in double precision and check the SiTU value
    // against its stored Q8 reconstruction within one half quantization step.
    for(int g=0;g<16;++g)for(int j=0;j<16;++j){
        int slot=(j+g)%8, ex=selected[g*8+slot], ch=(g*97+j*61)%1152;
        double xy[2]={};
        for(int half=0;half<2;++half){
            size_t row=(ex*2304+2*ch+half)*size_t(2304);
            for(int i=0;i<2304;++i){
                auto p=row+i; unsigned code=(raw[p/2]>>(4*(p%2)))&15;
                if(code!=0&&code!=7&&code!=15)return 5;
                unsigned sc=scale[p/16];double unit=std::ldexp(1.+double(sc%8)/8.,int(sc/8)-7);
                xy[half]+=(code==7?6:code==15?-6:0)*unit*double(act[i])*xs[i/16];
            }
            xy[half]*=amplitude[ex]/2688.;
        }
        double f=4*std::tanh(xy[0]*.25)/(1+std::exp(-xy[0]))*25*std::tanh(xy[1]*.04);
        auto pos=g*channels+slot*1152+ch;
        double step=new_s[pos/16], approx=new_q[pos]*step;
        if(std::abs(f-approx)>0.501*step+2.e-4){
            std::fprintf(stderr,"reference mismatch at %zu %.9g %.9g step %.9g\n",pos,f,approx,step);
            return 5;
        }
    }
    for(int c=0;c<9;++c)for(int l=0;l<4;++l){
        bool trial=(l==1||l==2)^(c%2!=0);
        std::printf("%d,%d,%s,%.6f\n",c,l,trial?"two":"four",run(trial));
    }
    RT(cudaEventDestroy(begin));RT(cudaEventDestroy(end));DR(cuModuleUnload(mod));
    for(void* p:std::vector<void*>{dp,dt,ds,da,dx,dscale,dids,dq,dsout})RT(cudaFree(p));
}
