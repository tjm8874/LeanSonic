#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "inkling2_sm120.cu"

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
        std::exit(2);
    }
}

template<class T> static T* upload(const std::vector<T>& v) {
    T* out;
    check(cudaMalloc(&out, v.size()*sizeof(T)));
    check(cudaMemcpy(out,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
    return out;
}

template<int Tile>
__global__ void verify(const int* counts, const int4* expected, unsigned* errors) {
    __shared__ int4 shared_tile;
    int expert = -1, first = 0, count = 0;
    size_t sorted = 0;
    const bool found = resolve_flat_expert_tile<Tile>(
        counts + 128*blockIdx.x, shared_tile, expert, first, count, sorted);
    const int4 ref = expected[blockIdx.x*gridDim.y+blockIdx.y];
    const bool bad = found != (ref.x >= 0) ||
        (found && (ref.x!=expert || ref.y!=first || ref.z!=count || size_t(ref.w)!=sorted));
    const int mismatches = __syncthreads_count(bad);
    if (threadIdx.x==0 && mismatches) atomicAdd(errors,unsigned(mismatches));
}

template<int Tile>
static void run() {
    using Counts = std::array<int,128>;
    std::vector<Counts> cases(1);
    for(int expert : {0,1,3,4,31,32,63,64,123,124,127}) {
        for (int count : {1,Tile-1,Tile,Tile+1,2*Tile-1,2*Tile,2048}) {
            Counts c{}; c[expert]=count; cases.push_back(c);
        }
    }
    for(unsigned seed=0; seed<24; ++seed) {
        Counts c{};
        for(unsigned i=0; i<128; ++i) {
            unsigned x=(i+17)*(seed+20903u)*2654435761u;
            c[i]=(x>>29)<(seed%8) ? 0 : int((x^(x>>10))%300);
        }
        cases.push_back(c);
    }
    Counts uniform{};uniform.fill(128); cases.push_back(uniform);
    // Exercise a row offset beyond 16 bits as well as the current 2K prefill shapes.
    uniform.fill(640);cases.push_back(uniform);
    int max_tiles=0;
    for(const auto& c:cases) {
        int n=0;for(int rows:c)n+=(rows+Tile-1)/Tile;
        max_tiles=std::max(n,max_tiles);
    }
    const int pitch=max_tiles+3;
    std::vector<int> counts;
    std::vector<int4> wanted(cases.size()*pitch, make_int4(-1,0,0,0));
    for(size_t k=0;k<cases.size();++k) {
        int row=0, tile=0;
        for(int e=0;e<128;++e) {
            int n=cases[k][e];counts.push_back(n);
            for(int first=0;first<n;first+=Tile)
                wanted[k*pitch+tile++]=make_int4(e,first,n,row+first);
            row+=n;
        }
    }
    int* device_counts=upload(counts);
    int4* expected=upload(wanted);
    unsigned* errors=upload(std::vector<unsigned>{0});
    for(int threads:{256,384,512}) {
        verify<Tile><<<dim3(cases.size(), pitch),threads>>>(device_counts,expected,errors);
        check(cudaGetLastError());
    }
    unsigned mismatches=1;
    check(cudaMemcpy(&mismatches,errors,sizeof(mismatches),cudaMemcpyDeviceToHost));
    if (mismatches) { std::fprintf(stderr,"%u thread mismatches\n",mismatches);std::exit(3); }
    check(cudaFree(errors));check(cudaFree(expected));check(cudaFree(device_counts));
    std::printf("tile=%d count_vectors=%zu grid_y=%d block_sizes=256,384,512 mismatches=%u\n",
        Tile,cases.size(),pitch,mismatches);
}
int main() { run<128>();run<160>(); }
