#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr size_t groups = 51200, experts = 512;
struct Layout {
  uint64_t high_bytes = 0, indexed_experts = 0;
};
static size_t align128(size_t n) { return (n + 127) / 128 * 128; }
// Each declared projection has the exponent span required by production Z24.
static uint8_t code(uint16_t x, uint16_t base) {
  uint16_t exp = (x >> 7) & 255;
  if (!exp) {
    if (x & 127)
      throw std::runtime_error("subnormal");
    return 0;
  }
  if (exp < base || exp - base >= 15)
    throw std::runtime_error("exponent span");
  return static_cast<uint8_t>(exp - base + 1);
}
int main(int argc, char **argv) {
  if (argc != 2 && argc != 3)
    return 2;
  if (argc == 3 && !std::filesystem::create_directory(argv[2]))
    throw std::runtime_error("choose a fresh directory");
  uint64_t written = 0;
  std::ifstream spec(argv[1]);
  std::string file, bfile;
  uint64_t so, bo;
  std::array<Layout, 5> results{};
  std::array<uint64_t, 257> histogram{};
  size_t tensors = 0;
  while (spec >> file >> so >> bfile >> bo) {
    std::ifstream a(file, std::ios::binary), b(bfile, std::ios::binary);
    std::vector<uint16_t> scale(groups * experts), bias(groups * experts);
    if (!a.seekg(so).read(reinterpret_cast<char *>(scale.data()),
                          scale.size() * 2) ||
        !b.seekg(bo).read(reinterpret_cast<char *>(bias.data()),
                          bias.size() * 2))
      throw std::runtime_error("short read");
    auto base = [](auto &words) {
      uint16_t mn = 255;
      for (auto w : words) {
        auto e = (w >> 7) & 255;
        if (e)
          mn = std::min(mn, static_cast<uint16_t>(e));
      }
      return mn;
    };
    auto sb = base(scale), bb = base(bias);
    std::string path;
    std::fstream physical;
    if (argc == 3) {
      path = std::string(argv[2]) + "/projection-" + std::to_string(tensors) +
             ".bin";
      physical.open(path, std::ios::in | std::ios::out | std::ios::binary |
                              std::ios::trunc);
      if (!physical)
        throw std::runtime_error("cannot open output");
    }
    for (size_t e = 0; e < experts; e++) {
      std::array<int, 256> map;
      map.fill(-1);
      std::array<uint8_t, 256> dict{};
      size_t count = 0;
      std::array<uint8_t, groups> indices{};
      for (size_t i = 0; i < groups; i++) {
        auto k = e * groups + i;
        auto hi =
            static_cast<uint8_t>((code(bias[k], bb) << 4) | code(scale[k], sb));
        if (map[hi] < 0) {
          map[hi] = count;
          dict[count++] = hi;
        }
        indices[i] = static_cast<uint8_t>(map[hi]);
      }
      histogram[count]++;
      if (argc == 3) {
        if (count > 16)
          throw std::runtime_error("four-bit dictionary overflow");
        std::array<uint8_t, 128> table{};
        std::copy_n(dict.begin(), count, table.begin());
        std::array<uint8_t, groups / 2> payload{};
        for (size_t i = 0; i < groups; i += 2)
          payload[i / 2] = indices[i] | (indices[i + 1] << 4);
        if (!physical.write(reinterpret_cast<const char *>(table.data()),
                            table.size()) ||
            !physical.write(reinterpret_cast<const char *>(payload.data()),
                            payload.size()))
          throw std::runtime_error("write");
        written += table.size() + payload.size();
      }
      // Fixed bits have deterministic offsets within each expert. Compare a
      // 128-byte aligned table/index and per-expert descriptor vs the existing
      // raw Z24 high bytes.
      for (unsigned bits = 3; bits <= 7; bits++) {
        auto &r = results[bits - 3];
        size_t n = align128(1u << bits) + align128((groups * bits + 7) / 8);
        if (count > (1u << bits) || n >= groups) {
          r.high_bytes += groups + 8;
          continue;
        }
        r.high_bytes += n + 8;
        r.indexed_experts++;
      }
      for (size_t i = 0; i < groups; i++) {
        auto k = e * groups + i;
        auto hi = dict[indices[i]];
        auto rebuild = [](uint16_t w, uint8_t hi, uint16_t base) {
          return static_cast<uint16_t>((w & 0x807f) |
                                       ((hi ? base + hi - 1 : 0) << 7));
        };
        if (rebuild(scale[k], hi & 15, sb) != scale[k] ||
            rebuild(bias[k], hi >> 4, bb) != bias[k])
          throw std::runtime_error("pair roundtrip");
      }
    }
    if (argc == 3) {
      physical.close();
      std::ifstream saved(path, std::ios::binary);
      for (size_t e = 0; e < experts; e++) {
        std::array<uint8_t, 128> table{};
        std::array<uint8_t, groups / 2> payload{};
        if (!saved.read(reinterpret_cast<char *>(table.data()), table.size()) ||
            !saved.read(reinterpret_cast<char *>(payload.data()),
                        payload.size()))
          throw std::runtime_error("readback");
        for (size_t i = 0; i < groups; i++) {
          auto k = e * groups + i;
          auto hi = table[(payload[i / 2] >> ((i & 1) * 4)) & 15];
          if (hi != ((code(bias[k], bb) << 4) | code(scale[k], sb)))
            throw std::runtime_error("stored upper bits mismatch");
        }
      }
      if (saved.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("trailing bytes");
    }
    ++tensors;
  }
  if (tensors != 144)
    throw std::runtime_error("missing projections");
  std::printf("{\"verified_physical_high_bytes\":%llu,\"projections\":%zu,"
              "\"experts_per_projection\":%zu,\"groups_per_expert\":%zu,"
              "\"original_high_bytes\":%zu,\"layouts\":[",
              (unsigned long long)written, tensors, experts, groups,
              tensors * experts * groups);
  for (unsigned i = 0; i < 5; i++)
    std::printf("%s{\"bits\":%u,\"indexed_experts\":%llu,\"high_bytes\":%llu}",
                i ? "," : "", i + 3,
                (unsigned long long)results[i].indexed_experts,
                (unsigned long long)results[i].high_bytes);
  std::printf("],\"palette_size_histogram\":{");
  bool first = true;
  for (unsigned i = 1; i <= 256; i++)
    if (histogram[i]) {
      std::printf("%s\"%u\":%llu", first ? "" : ",", i,
                  (unsigned long long)histogram[i]);
      first = false;
    }
  std::puts("}}");
}
