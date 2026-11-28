// Exercise the shipped cubin with exactly sized buffers. Build with -lcuda.
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "CUDA error %d (%s)\n", int(result), message ? message : "unknown");
    std::exit(1);
}

static float as_float(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

static float bf16_round(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof word);
    return as_float((word + 0x7fff + ((word >> 16) & 1)) >> 16);
}

template<class T> static CUdeviceptr upload(const std::vector<T>& data) {
    CUdeviceptr p;
    check(cuMemAlloc(&p, data.size() * sizeof(T)));
    check(cuMemcpyHtoD(p, data.data(), data.size() * sizeof(T)));
    return p;
}

static void exercise(CUfunction fn, int rows, int cols, int batch, bool red, bool staged = false, bool tiled = false) {
    std::vector<int8_t> weights(size_t(rows) * cols);
    std::vector<uint16_t> scales(weights.size() / 32);
    std::vector<float> inputs(size_t(batch) * cols);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = int((i * 61 + i / 13) % 255) - 127;
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = i % 17 ? 0x3b80 + (i * 37 % 512) : 0;
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = (int((i * 17 + i / 37) % 37) - 18) / 16.0f;
    std::vector<float> outputs(size_t(batch) * rows - (red ? 1 : 0), NAN);
    auto w = upload(weights), s = upload(scales), x = upload(inputs), y = upload(outputs);
    void* args[] = {&w, &s, &x, &y, &rows, &cols, &batch};
    unsigned scratch = staged ? (cols / 32) * 9 * 16 : 0;
    check(cuLaunchKernel(fn, (rows + 7) / 8, tiled ? (batch + 3) / 4 : batch, 1, 256, 1, 1, scratch, nullptr, args, nullptr));
    CUresult result = cuCtxSynchronize();
    if (red) return; // memcheck owns the one-float-short fault and its exit code.
    check(result);
    check(cuMemcpyDtoH(outputs.data(), y, outputs.size() * sizeof(float)));
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            double expected = 0;
            for (int c = 0; c < cols; ++c) {
                size_t i = size_t(r) * cols + c;
                expected += double(bf16_round(weights[i] * as_float(scales[i / 32]))) * inputs[size_t(b) * cols + c];
            }
            float actual = outputs[size_t(b) * rows + r];
            double difference = std::abs(actual - expected);
            if (!std::isfinite(actual) || difference > 1e-3) {
                std::fprintf(stderr, "row %d batch %d: actual %.9g reference %.9g\n", r, b, actual, expected);
                std::exit(1);
            }
            worst = std::fmax(worst, difference);
        }
    }
    std::printf("rows=%d cols=%d batch=%d max_abs_err=%.9g\n", rows, cols, batch, worst);
    for (auto ptr : {w, s, x, y}) check(cuMemFree(ptr));
}

// Check the fixed-shape nonlinear mixer against scalar reference dots. Include both
// final-mixer and intermediate-mixer layouts, and batched token strides.
static double reference_dot(const std::vector<int8_t>& weights, const std::vector<uint16_t>& scales,
                            const std::vector<float>& x, int row, int cols, int token) {
    double sum = 0;
    for (int i = 0; i < cols; ++i) {
        const size_t offset = size_t(row) * cols + i;
        sum += double(bf16_round(weights[offset] * as_float(scales[offset / 32]))) * x[size_t(token) * cols + i];
    }
    return sum;
}

static void check_close(float actual, double expected, double& worst) {
    const double diff = std::abs(actual - expected);
    if (!std::isfinite(actual) || diff > 0.0004 * (1.0 + std::abs(expected))) {
        std::fprintf(stderr, "mixer output %.9g expected %.9g\n", actual, expected);
        std::exit(1);
    }
    worst = std::fmax(worst, diff);
}

static void exercise_hc(CUmodule module, const char* fault, int rows, int batch) {
    CUfunction down, up;
    check(cuModuleGetFunction(&down, module, "q38_hc_q8_down_inject"));
    check(cuModuleGetFunction(&up, module, "q38_hc_q8_up_mix"));
    constexpr int dim = 2560, wide = 4 * dim, lowdim = 320;
    std::vector<int8_t> down_w(size_t(rows) * wide), up_w(size_t(wide) * lowdim);
    std::vector<uint16_t> down_s(down_w.size()/32), up_s(up_w.size()/32);
    for (auto* weights : {&down_w, &up_w})
        for (size_t i = 0; i < weights->size(); ++i)
            (*weights)[i] = int((i * 37 + i / 23) % 255) - 127;
    for (auto* scales : {&down_s, &up_s})
        for (size_t i = 0; i < scales->size(); ++i)
            (*scales)[i] = i % 19 ? 0x3900 + (i * 17 % 384) : 0;
    std::vector<float> normed(size_t(batch)*wide);
    for (size_t i = 0; i < normed.size(); ++i) normed[i] = (int((i*13+i/89)%31)-15)/16.0f;
    const bool short_low = !std::strcmp(fault, "hc-red-down");
    const bool short_mix = !std::strcmp(fault, "hc-red-up");
    std::vector<float> low(size_t(batch)*lowdim - short_low, NAN);
    std::vector<float> inj(size_t(batch)*4, NAN), mixed(size_t(batch)*dim - short_mix, NAN);
    auto dw=upload(down_w), ds=upload(down_s), uw=upload(up_w), us=upload(up_s);
    auto n=upload(normed), l=upload(low), j=upload(inj), m=upload(mixed);
    void* down_args[] = {&dw, &ds, &n, &l, &j, &rows};
    check(cuLaunchKernel(down, (rows+7)/8, batch, 1, 256, 1, 1, 0, nullptr, down_args, nullptr));
    CUresult result = cuCtxSynchronize();
    if (short_low) return;
    check(result);
    void* up_args[] = {&uw, &us, &l, &n, &m};
    check(cuLaunchKernel(up, dim/8, batch, 1, 256, 1, 1, 0, nullptr, up_args, nullptr));
    result = cuCtxSynchronize();
    if (short_mix) return;
    check(result);
    check(cuMemcpyDtoH(low.data(), l, low.size()*sizeof(float)));
    check(cuMemcpyDtoH(inj.data(), j, inj.size()*sizeof(float)));
    check(cuMemcpyDtoH(mixed.data(), m, mixed.size()*sizeof(float)));
    double worst_low=0, worst_inj=0, worst_mix=0;
    for (int b=0; b<batch; ++b) {
        for (int r=0; r<rows; ++r) {
            double dot=reference_dot(down_w, down_s, normed, r, wide, b);
            if (r<lowdim) {
                double z=dot*0.25;
                check_close(low[b*lowdim+r], z/(1.0+std::exp(-z)), worst_low);
            } else {
                check_close(inj[b*4+r-lowdim], dot, worst_inj);
            }
        }
        if (rows==320)
            for (int i=0; i<4; ++i) if (!std::isnan(inj[b*4+i])) std::exit(1);
        for (int x=0; x<dim; ++x) {
            double value=0;
            for (int k=0; k<4; ++k) {
                int r=k*dim+x;
                value+=normed[b*wide+r]/(1.0+std::exp(-reference_dot(up_w,up_s,low,r,lowdim,b)));
            }
            check_close(mixed[b*dim+x],value*0.25,worst_mix);
        }
    }
    std::printf("rows=%d batch=%d low=%.8g inj=%.8g mixed=%.8g\n",rows,batch,worst_low,worst_inj,worst_mix);
    for (auto ptr : {dw,ds,uw,us,n,l,j,m}) check(cuMemFree(ptr));
}

static void check_embedding(CUmodule module, bool red, int batch) {
    constexpr int rows = 11, cols = 2560;
    std::vector<int8_t> weights(rows * cols);
    std::vector<uint16_t> scales(weights.size()/32);
    std::vector<int> tokens(batch);
    for (size_t i=0; i<weights.size(); ++i) weights[i]=int((i*101+i/53)%255)-127;
    for (size_t i=0; i<scales.size(); ++i) scales[i]=i%13 ? 0x3a80+(i*5%512) : 0;
    for (int i=0; i<batch; ++i) tokens[i]=i%3 ? 0 : rows-1;
    std::vector<float> out(size_t(batch)*4*cols-red,NAN);
    auto w=upload(weights), sc=upload(scales), t=upload(tokens), o=upload(out);
    CUdeviceptr zero=0;
    void* args[]={&w,&sc,&zero,&t,&o};
    CUfunction fn;
    check(cuModuleGetFunction(&fn,module,"q38_embed_q8_hc4"));
    check(cuLaunchKernel(fn,cols/256,batch,1,256,1,1,0,nullptr,args,nullptr));
    CUresult result=cuCtxSynchronize();
    if (red) return;
    check(result);
    check(cuMemcpyDtoH(out.data(),o,out.size()*sizeof(float)));
    for (int b=0; b<batch; ++b) for (int k=0; k<4; ++k) for (int d=0; d<cols; ++d) {
        int index=tokens[b]*cols+d;
        float expected=bf16_round(weights[index]*as_float(scales[index/32]));
        if (out[(b*4+k)*cols+d]!=expected) std::exit(1);
    }
    std::printf("embedding batch=%d exact=%zu\n",batch,out.size());
    for (auto ptr:{w,sc,t,o}) check(cuMemFree(ptr));
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    check(cuInit(0));
    CUdevice device;
    check(cuDeviceGet(&device, 0));
    CUcontext context;
    check(cuDevicePrimaryCtxRetain(&context, device));
    check(cuCtxSetCurrent(context));
    CUmodule module;
    check(cuModuleLoad(&module, argv[1]));
    if (!std::strcmp(argv[2], "tile-green") || !std::strcmp(argv[2], "tile-red")) {
        CUfunction fn;
        check(cuModuleGetFunction(&fn, module, "q38_dense_q8_gemm_b4"));
        if (!std::strcmp(argv[2], "tile-red")) exercise(fn, 9, 160, 5, true, false, true);
        else for (int batch : {1,2,3,4,5,7,8,17,32,64})
            for (int cols : {32,160,2560,10240}) exercise(fn, 9, cols, batch, false, false, true);
        return 0;
    }
    if (!std::strcmp(argv[2], "embed-green") || !std::strcmp(argv[2], "embed-red")) {
        bool red=!std::strcmp(argv[2], "embed-red");
        check_embedding(module,red,1);
        if (!red) check_embedding(module,false,17);
        return 0;
    }
    if (!std::strcmp(argv[2], "shared-green") || !std::strcmp(argv[2], "shared-red")) {
        CUfunction fn;
        check(cuModuleGetFunction(&fn, module, "q38_shared_q8_projection"));
        bool red=!std::strcmp(argv[2], "shared-red");
        if (red) exercise(fn,9,640,2,true,true);
        else for (int cols:{640,2560}) for (int batch:{1,2,7,32,64}) exercise(fn,17,cols,batch,false,true);
        return 0;
    }
    if (!std::strcmp(argv[2], "hc-green")) {
        for (int rows : {320,324})
            for (int batch : {1,2,5,8}) exercise_hc(module, argv[2], rows, batch);
        check(cuModuleUnload(module));
        check(cuDevicePrimaryCtxRelease(device));
        return 0;
    }
    if (!std::strcmp(argv[2], "hc-red-down") || !std::strcmp(argv[2], "hc-red-up")) {
        exercise_hc(module, argv[2], 324, 2);
        return 0;
    }
    if (std::strcmp(argv[2], "green") && std::strcmp(argv[2], "red")) return 2;
    CUfunction fn;
    check(cuModuleGetFunction(&fn, module, "q38_dense_q8_gemv"));
    if (!std::strcmp(argv[2], "red")) {
        exercise(fn, 9, 160, 2, true);
        return 0;
    }
    for (int batch : {1, 2, 3, 4, 8, 32, 64}) {
        for (int cols : {32, 160, 2560, 10240}) exercise(fn, 9, cols, batch, false);
    }
    check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(device));
}
