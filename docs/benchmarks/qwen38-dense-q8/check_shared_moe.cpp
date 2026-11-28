// Verify shared Q8 accesses inside all eight actual routed MoE kernel entry points.
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char* msg; cuGetErrorString(result, &msg);
        std::fprintf(stderr, "%s\n", msg); std::exit(1);
    }
}

template<class T> static CUdeviceptr upload(const std::vector<T>& data) {
    CUdeviceptr ptr;
    check(cuMemAlloc(&ptr, data.size()*sizeof(T)));
    check(cuMemcpyHtoD(ptr, data.data(), data.size()*sizeof(T)));
    return ptr;
}

static void equal(float actual, double expected) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > 2e-5 * (1.0+std::abs(expected))) {
        std::fprintf(stderr, "expected %.9g, observed %.9g\n", expected, actual);
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::string name=argv[2], mode=argv[3];
    bool up=name.find("_up_")!=std::string::npos, grouped=name.find("_grouped_")!=std::string::npos;
    bool bn=name.ends_with("_bn"), z24=name.find("_z24")!=std::string::npos;
    if (mode!="green" && mode!="red") return 2;
    const bool red=mode=="red";
    int batch=grouped ? 9 : (bn ? 3 : 1);
    int rows=up ? 1281 : 2560, cols=up ? 2560 : 640;
    check(cuInit(0));
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
    check(cuDeviceGet(&dev,0)); check(cuDevicePrimaryCtxRetain(&ctx,dev)); check(cuCtxSetCurrent(ctx));
    check(cuModuleLoad(&mod,argv[1])); check(cuModuleGetFunction(&fn,mod,name.c_str()));
    // All active routed IDs are zero. One valid expert record suffices and produces zero.
    auto routed=upload(std::vector<uint8_t>(819200,0));
    auto metadata=upload(std::vector<uint32_t>(25600,0));
    auto shared=upload(std::vector<int8_t>(size_t(rows)*cols-(red?32:0),1));
    auto scales=upload(std::vector<uint16_t>(size_t(rows)*cols/32,0x3b80)); // 1/256
    CUdeviceptr bias=0;
    auto ids=upload(std::vector<int>(batch*10,0));
    auto offsets=upload(std::vector<int>(513,0));
    auto x=upload(std::vector<float>(size_t(batch)*cols*(up?1:11),0.125f));
    auto wts=upload(std::vector<float>(batch*11,0.5f));
    int outwidth=up ? 640 : 2560;
    bool has_slots=up || grouped;
    std::vector<float> output(size_t(batch)*outwidth*(has_slots?11:1),0);
    auto out=upload(output);
    uint32_t bases=0x76727672u; int compact=0;
    std::vector<void*> args={&routed,&metadata,&metadata};
    if (up) args.insert(args.end(),{&routed,&metadata,&metadata});
    if (z24) args.push_back(&bases);
    args.insert(args.end(),{&shared,&scales,&bias,&x});
    if (grouped) {
        args.insert(args.end(),{&offsets,&ids});
        if (up) args.insert(args.end(),{&batch,&wts});
        else args.insert(args.end(),{&wts,&batch});
    } else {
        args.push_back(&ids);
        if (!z24) args.push_back(&compact);
        args.push_back(&wts);
    }
    args.push_back(&out);
    int gx=up ? (grouped?80:160) : (grouped?40:(z24?80:320));
    int gy=grouped ? 513 : (up?11*batch:batch);
    check(cuLaunchKernel(fn,gx,gy,1,up&&grouped?512:256,1,1,0,nullptr,args.data(),nullptr));
    CUresult result=cuCtxSynchronize();
    if (red) return 0;
    check(result);
    check(cuMemcpyDtoH(output.data(),out,output.size()*sizeof(float)));
    std::vector<float> scores(batch*11);
    check(cuMemcpyDtoH(scores.data(),wts,scores.size()*sizeof(float)));
    double dot=cols*0.125/256.0;
    double sigmoid=1.0/(1.0+std::exp(-dot));
    for (int b=0;b<batch;++b) {
        if (up) equal(scores[b*11+10],sigmoid);
        size_t start=outwidth*(has_slots ? b*11+10 : b);
        for (int d=0;d<outwidth;++d) equal(output[start+d],up?dot*dot*sigmoid:0.5*dot);
    }
    std::printf("%s checked %d shared outputs\n",argv[2],batch*outwidth);
}
