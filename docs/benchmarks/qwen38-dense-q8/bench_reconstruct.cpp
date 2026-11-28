// Alternate the original and candidate functions on identical buffers and CUDA events.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <random>
#include <string>
#include <vector>

static void check(CUresult rc) {
    if (rc != CUDA_SUCCESS) {
        const char *reason;
        cuGetErrorString(rc, &reason);
        std::fprintf(stderr, "CUDA %d: %s\n", rc, reason);
        std::exit(1);
    }
}

template <typename T> static CUdeviceptr upload(const std::vector<T> &x) {
    CUdeviceptr p;
    check(cuMemAlloc(&p, x.size() * sizeof(T)));
    check(cuMemcpyHtoD(p, x.data(), x.size() * sizeof(T)));
    return p;
}

struct Shape {
    const char *op;
    int rows, cols, batch;
};

static void measure(Shape shape, std::array<CUmodule, 2> modules, bool tiled) {
    int rows = shape.rows, cols = shape.cols, batch = shape.batch;
    bool shared = !std::strcmp(shape.op, "q38_shared_q8_projection");
    bool mixer = !std::strcmp(shape.op, "q38_hc_q8_up_mix");
    bool lower = !std::strcmp(shape.op, "q38_hc_q8_down_inject");
    bool tiled_projection = tiled && batch > 1 && !std::strcmp(shape.op, "q38_dense_q8_gemv");
    size_t count = size_t(rows) * cols;
    std::mt19937 gen(17);
    std::vector<uint32_t> codes(count / 4);
    for (auto &x : codes)
        x = gen();
    std::vector<uint16_t> scales(count / 32);
    for (auto &x : scales)
        x = 0x3880 + gen() % 1024;
    std::vector<float> input(size_t(batch) * cols), norm(size_t(batch) * 10240);
    for (auto &x : input)
        x = float(int(gen() % 4097) - 2048) / 2048;
    for (auto &x : norm)
        x = float(int(gen() % 4097) - 2048) / 2048;
    CUdeviceptr w = upload(codes), s = upload(scales), x = upload(input), n = upload(norm);
    int real_rows = mixer ? 2560 : (lower ? 320 : rows);
    size_t nout = size_t(real_rows) * batch;
    std::array<CUfunction, 2> f;
    std::array<CUdeviceptr, 2> out, inj;
    for (int k = 0; k < 2; ++k) {
        check(cuModuleGetFunction(
            &f[k], modules[k], (k == 1 && tiled_projection) ? "q38_dense_q8_gemm_b4" : shape.op));
        check(cuMemAlloc(&out[k], nout * sizeof(float)));
        check(cuMemAlloc(&inj[k], batch * 4 * sizeof(float)));
    }
    auto launch = [&](int k) {
        void *proj_args[] = {&w, &s, &x, &out[k], &rows, &cols, &batch};
        void *up_args[] = {&w, &s, &x, &n, &out[k]};
        void *down_args[] = {&w, &s, &x, &out[k], &inj[k], &rows};
        check(cuLaunchKernel(f[k], ((mixer ? 2560 : rows) + 7) / 8,
                             k == 1 && tiled_projection ? (batch + 3) / 4 : batch, 1, 256, 1, 1,
                             shared ? cols / 32 * 9 * 16 : 0, nullptr,
                             mixer ? up_args : (lower ? down_args : proj_args), nullptr));
    };
    for (int k : {0, 1})
        launch(k);
    check(cuCtxSynchronize());
    auto same = [&](const std::array<CUdeviceptr, 2> &p, size_t items) {
        std::vector<uint32_t> a(items), b(items);
        check(cuMemcpyDtoH(a.data(), p[0], items * 4));
        check(cuMemcpyDtoH(b.data(), p[1], items * 4));
        if (a != b) {
            std::fprintf(stderr, "%s %dx%d B%d unequal FP32 results\n", shape.op, rows, cols,
                         batch);
            std::exit(2);
        }
    };
    same(out, nout);
    if (lower && rows == 324)
        same(inj, size_t(batch) * 4);
    CUevent start, stop;
    check(cuEventCreate(&start, 0));
    check(cuEventCreate(&stop, 0));
    int repeats = count * batch > (size_t(1) << 29) ? 4 : 16;
    for (int round = 0; round < 8; ++round) {
        for (int order = 0; order < 2; ++order) {
            int k = (order + round) % 2;
            launch(k);
            check(cuEventRecord(start, nullptr));
            for (int rep = 0; rep < repeats; ++rep)
                launch(k);
            check(cuEventRecord(stop, nullptr));
            check(cuEventSynchronize(stop));
            float ms;
            check(cuEventElapsedTime(&ms, start, stop));
            std::printf("%s\t%d\t%d\t%d\t%d\t%d\t%.6f\n", shape.op, rows, cols, batch, round, k,
                        ms * 1000 / repeats);
        }
    }
    std::fflush(stdout);
    check(cuEventDestroy(start));
    check(cuEventDestroy(stop));
    for (auto p : {w, s, x, n, out[0], out[1], inj[0], inj[1]})
        check(cuMemFree(p));
}

int main(int argc, char **argv) {
    if (argc != 3 && (argc != 4 || std::strcmp(argv[3], "tiled")))
        return 2;
    const bool tiled = argc == 4;
    check(cuInit(0));
    CUdevice dev;
    CUcontext ctx;
    check(cuDeviceGet(&dev, 0));
    check(cuDevicePrimaryCtxRetain(&ctx, dev));
    check(cuCtxSetCurrent(ctx));
    std::array<CUmodule, 2> modules;
    for (int k = 0; k < 2; ++k)
        check(cuModuleLoad(&modules[k], argv[k + 1]));
    std::puts("kernel\trows\tcols\tbatch\tround\tarm\tmicroseconds");
    for (int batch : {1, 8})
        for (Shape shape : {Shape{"q38_dense_q8_gemv", 16480, 2560, batch},
                            Shape{"q38_dense_q8_gemv", 2560, 6144, batch},
                            Shape{"q38_dense_q8_gemv", 248320, 2560, batch},
                            Shape{"q38_shared_q8_projection", 1281, 2560, batch},
                            Shape{"q38_shared_q8_projection", 2560, 640, batch},
                            Shape{"q38_hc_q8_down_inject", 324, 10240, batch},
                            Shape{"q38_hc_q8_up_mix", 10240, 320, batch}})
            measure(shape, modules, tiled);
    if (tiled)
        for (int batch : {2, 3, 5, 17})
            measure({"q38_dense_q8_gemv", 257, 2560, batch}, modules, true);
}
