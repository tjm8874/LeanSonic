#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr unsigned groups = 51200, experts = 512, stride = 128 + groups / 2;
constexpr size_t raw_size = experts * groups, packed_size = experts * stride;
void check(cudaError_t err) {
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));
}
__global__ void expand(const uint8_t *packed, uint8_t *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= raw_size)
    return;
  unsigned expert = i / groups, local = i % groups;
  const uint8_t *rec = packed + expert * stride;
  unsigned index = (rec[128 + local / 2] >> ((local & 1) * 4)) & 15;
  out[i] = rec[index];
}
size_t allocated(const size_t bytes) {
  std::vector<void *> ptrs(144);
  size_t before, after, total;
  check(cudaMemGetInfo(&before, &total));
  for (auto &p : ptrs)
    check(cudaMalloc(&p, bytes));
  check(cudaDeviceSynchronize());
  check(cudaMemGetInfo(&after, &total));
  for (auto p : ptrs)
    check(cudaFree(p));
  return before - after;
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  check(cudaSetDevice(0));
  check(cudaFree(nullptr));
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, 0));
  auto baseline = allocated(raw_size), packed = allocated(packed_size);
  uint8_t *ip, *op;
  check(cudaMalloc(&ip, packed_size));
  check(cudaMalloc(&op, raw_size));
  std::vector<uint8_t> host(packed_size), decoded(raw_size);
  size_t checked = 0;
  for (unsigned proj = 0; proj < 144; proj++) {
    std::ifstream f(std::string(argv[1]) + "/projection-" +
                        std::to_string(proj) + ".bin",
                    std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(host.data()), host.size()) ||
        f.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("bad packed projection");
    check(cudaMemcpy(ip, host.data(), packed_size, cudaMemcpyHostToDevice));
    expand<<<(raw_size + 255) / 256, 256>>>(ip, op);
    check(cudaGetLastError());
    check(cudaMemcpy(decoded.data(), op, raw_size, cudaMemcpyDeviceToHost));
    for (unsigned e = 0; e < experts; e++) {
      const uint8_t *rec = host.data() + e * stride;
      for (unsigned i = 0; i < groups / 2; i++) {
        auto b = rec[128 + i];
        auto at = e * groups + i * 2;
        if (decoded[at] != rec[b & 15] || decoded[at + 1] != rec[b >> 4])
          throw std::runtime_error("device output mismatch");
      }
    }
    checked += raw_size;
  }
  check(cudaFree(ip));
  check(cudaFree(op));
  std::printf("{\"gpu\":\"%s\",\"verified_groups\":%zu,\"raw_payload_bytes\":%"
              "zu,\"packed_payload_bytes\":%zu,\"raw_device_allocation_delta\":"
              "%zu,\"packed_device_allocation_delta\":%zu}\n",
              prop.name, checked, 144 * raw_size, 144 * packed_size, baseline,
              packed);
}
