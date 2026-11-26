// c++ -O2 -I/usr/local/cuda/include kernels/tests/inkling_expert_pages.cpp -lcuda -o pages
// ./pages candidate.cubin
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "CUDA error %d: %s\n", result, message ? message : "unknown");
    std::exit(1);
}

static CUdeviceptr upload(const std::vector<int>& data) {
    CUdeviceptr ptr;
    check(cuMemAlloc(&ptr, data.size() * sizeof(int)));
    check(cuMemcpyHtoD(ptr, data.data(), data.size() * sizeof(int)));
    return ptr;
}

static void compare(CUdeviceptr ptr, const std::vector<int>& expected, const char* name, int batch) {
    std::vector<int> actual(expected.size());
    check(cuMemcpyDtoH(actual.data(), ptr, actual.size() * sizeof(int)));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s batch=%d offset=%zu expected=%d actual=%d\n",
                         name, batch, i, expected[i], actual[i]);
            std::exit(1);
        }
    }
}

static void test(CUfunction group, int batch, bool concentrated) {
    constexpr int experts = 128, page = 64, top_k = 8;
    std::vector<int> ids(batch * top_k), counts(experts), pairs((batch + 63) / page * experts * page, -1);
    auto tokens = pairs;
    auto d_pairs = upload(pairs), d_tokens = upload(tokens);
    auto d_counts = upload(std::vector<int>(experts, -1));
    for (int token = 0; token < batch; ++token) {
        for (int k = 0; k < top_k; ++k) {
            int pair = token * top_k + k;
            // One pattern crosses every page for the highest-ID experts. The other
            // mixes hot and rare experts, leaving partially populated pages.
            int expert = concentrated ? 127 - k : (k < 3 ? 125 + k : (token * 17 + k * 19) % 125);
            ids[pair] = expert;
            int rank = counts[expert]++;
            int offset = (rank / page * experts + expert) * page + rank % page;
            pairs[offset] = pair;
            tokens[offset] = token;
        }
    }
    auto d_ids = upload(ids);
    void* args[] = {&d_ids, &d_counts, &d_tokens, &d_pairs, &batch};
    check(cuLaunchKernel(group, 32, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
    compare(d_counts, counts, "counts", batch);
    compare(d_tokens, tokens, "token indices", batch);
    compare(d_pairs, pairs, "pair indices", batch);
    for (auto ptr : {d_ids, d_counts, d_tokens, d_pairs}) check(cuMemFree(ptr));
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    check(cuInit(0));
    CUdevice gpu;
    CUcontext context;
    check(cuDeviceGet(&gpu, 0));
    check(cuDevicePrimaryCtxRetain(&context, gpu));
    check(cuCtxSetCurrent(context));
    CUmodule module;
    CUfunction group;
    check(cuModuleLoad(&module, argv[1]));
    check(cuModuleGetFunction(&group, module, "ink2_build_expert_groups_wide_b"));
    for (int batch : {1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024}) {
        for (bool concentrated : {false, true}) test(group, batch, concentrated);
    }
    check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(gpu));
    std::puts("30 stable grouping and padding cases passed");
}
