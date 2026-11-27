#include <cuda.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void check(CUresult r) {
    if (r != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(r, &message);
        throw std::runtime_error(message ? message : "CUDA failure");
    }
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    constexpr int vocab = 200058;
    int tiles = (vocab + 2047) / 2048;
    check(cuInit(0));
    CUdevice device;
    CUcontext ctx;
    check(cuDeviceGet(&device, 0));
    check(cuDevicePrimaryCtxRetain(&ctx, device));
    check(cuCtxSetCurrent(ctx));
    CUmodule module;
    check(cuModuleLoad(&module, argv[1]));
    CUfunction old, stage, finish;
    check(cuModuleGetFunction(&old, module, "ink2_argmax_and_inc_pos"));
    check(cuModuleGetFunction(&stage, module, "ink2_argmax_partials"));
    check(cuModuleGetFunction(&finish, module, "ink2_argmax_finish_and_inc_pos"));
    CUdeviceptr full, partial, next, pos;
    check(cuMemAlloc(&full, vocab * 4 + tiles * 8));
    partial = full + vocab*4;
    check(cuMemAlloc(&next, 4));check(cuMemAlloc(&pos, 4));
    void *original[] = {&full, &next, &pos};
    void *first[] = {&full, &partial};
    void *last[] = {&partial, &tiles, &next, &pos};
    std::vector<float> logits(vocab);
    std::mt19937 rng(71243);
    for (int trial=0; trial<80; ++trial) {
        if (trial < 7) {
            std::fill(logits.begin(), logits.end(), trial==0 ? -0.0f : -30.0f);
            if(trial==1)logits.back()=1;
            if(trial==2)logits[2048]=logits[8191]=1;
            if(trial==3)logits[2047]=logits[2048]=1;
            if(trial==4)logits[255]=logits[256]=1;
            if(trial==5)logits[128]=logits.back()=1;
            if(trial==6)logits[0]=logits.back()=1;
        } else {
            for(auto &v:logits)v=static_cast<int>(rng()%65536)-32768;
        }
        int expected=static_cast<int>(std::max_element(logits.begin(), logits.end())-logits.begin());
        check(cuMemcpyHtoD(full, logits.data(), logits.size()*sizeof(float)));
        for (bool tiled : {false,true}) {
            const int initial=37+trial;
            check(cuMemcpyHtoD(pos,&initial,4));
            if (!tiled) check(cuLaunchKernel(old,1,1,1,256,1,1,0,nullptr,original,nullptr));
            else {
                check(cuLaunchKernel(stage,tiles,1,1,256,1,1,0,nullptr,first,nullptr));
                check(cuLaunchKernel(finish,1,1,1,256,1,1,0,nullptr,last,nullptr));
            }
            int result=-1,position=-1;
            check(cuMemcpyDtoH(&result,next,4));check(cuMemcpyDtoH(&position,pos,4));
            if(result!=expected || position!=initial+1)throw std::runtime_error("argmax or position mismatch");
        }
    }
    std::cout << "80/80 cases match CPU and original GPU argmax; exactly one position advance\n";
}
