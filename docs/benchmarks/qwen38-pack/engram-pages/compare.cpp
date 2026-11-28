// Compare ordinary Q4 weight/parameter planes with a single-page row gather on
// Linux.
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>

void require(bool ok, const std::string &what) {
  if (!ok)
    throw std::runtime_error(what);
}
struct Mapping {
  int fd;
  size_t size;
  uint8_t *data;
  explicit Mapping(const std::string &file) {
    fd = open(file.c_str(), O_RDONLY);
    require(fd >= 0, file);
    struct stat st{};
    require(fstat(fd, &st) == 0, "fstat");
    size = st.st_size;
    data = static_cast<uint8_t *>(
        mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    require(data != MAP_FAILED, "mmap");
  }
  Mapping(const Mapping &) = delete;
  ~Mapping() {
    munmap(data, size);
    close(fd);
  }
  size_t resident() const {
    std::vector<unsigned char> residency((size + 4095) / 4096);
    require(mincore(data, size, residency.data()) == 0, "mincore");
    return std::count_if(residency.begin(), residency.end(),
                         [](auto x) { return x & 1; });
  }
  void evict() {
    require(madvise(data, size, MADV_DONTNEED) == 0, "madvise");
    int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (rc)
      throw std::system_error(rc, std::generic_category(), "fadvise");
    require(madvise(data, size, MADV_RANDOM) == 0, "MADV_RANDOM");
  }
};
struct Row {
  std::array<uint8_t, 80> weights;
  std::array<uint16_t, 10> params;
  bool operator==(const Row &) const = default;
};
void decode(const uint8_t *plane, size_t row, const Mapping &dict,
            uint16_t *result) {
  size_t bit = row * 50;
  for (size_t j = 0; j < 5; ++j, bit += 10) {
    size_t i = bit / 8;
    unsigned window = plane[i] | (unsigned(plane[i + 1]) << 8);
    // Ten-bit indices begin at bit 0, 2, 4 or 6, so two bytes always suffice.
    unsigned code = (window >> (bit % 8)) & 1023;
    require(2 * code + 2 <= dict.size, "dictionary index");
    result[j] = uint16_t(dict.data[2 * code]) |
                (uint16_t(dict.data[2 * code + 1]) << 8);
  }
}
struct Data {
  Mapping weights, scales, biases, sd, bd, pages, tiles;
  size_t rows;
  explicit Data(const std::string &d)
      : weights(d + "/weights"), scales(d + "/scales"), biases(d + "/biases"),
        sd(d + "/scale-dict"), bd(d + "/bias-dict"), pages(d + "/pages"),
        tiles(d + "/tiles"), rows(weights.size / 80) {
    require(rows > 0 && weights.size == 80 * rows &&
                scales.size == (rows * 50 + 7) / 8 &&
                biases.size == scales.size && sd.size % 2 == 0 &&
                bd.size % 2 == 0 && tiles.size == ((rows + 3) / 4) * 370 &&
                pages.size == ((rows + 43) / 44) * 4096,
            "invalid fixture lengths");
  }
  Row read(size_t row, int layout) const {
    Row out;
    if (layout == 0) {
      std::memcpy(out.weights.data(), weights.data + row * 80, 80);
      decode(scales.data, row, sd, out.params.data());
      decode(biases.data, row, bd, out.params.data() + 5);
    } else {
      size_t count = layout == 1 ? 44 : 4;
      const auto *base = layout == 1 ? pages.data + (row / 44) * 4096
                                     : tiles.data + (row / 4) * 370;
      size_t slot = row % count;
      std::memcpy(out.weights.data(), base + slot * 80, 80);
      decode(base + count * 80, slot, sd, out.params.data());
      decode(base + count * 80 + count * 50 / 8, slot, bd,
             out.params.data() + 5);
    }
    return out;
  }
  size_t resident(int layout) const {
    if (layout == 0)
      return weights.resident() + scales.resident() + biases.resident();
    return layout == 1 ? pages.resident() : tiles.resident();
  }
  void evict(int layout) {
    if (layout == 1)
      pages.evict();
    else if (layout == 2)
      tiles.evict();
    else {
      weights.evict();
      scales.evict();
      biases.evict();
    }
  }
};
uint64_t bench(Data &d, int layout, bool cold, int seed,
               const std::vector<size_t> &ids) {
  if (cold)
    d.evict(layout);
  size_t before = d.resident(layout);
  if (cold)
    require(before == 0, "cold fixture has resident pages");
  rusage first{}, last{};
  require(getrusage(RUSAGE_SELF, &first) == 0, "getrusage");
  auto start = std::chrono::steady_clock::now();
  uint64_t sum = 0;
  for (auto i : ids) {
    auto r = d.read(i, layout);
    for (auto v : r.weights)
      sum += v;
    for (auto v : r.params)
      sum += v;
  }
  auto elapsed = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - start)
                     .count();
  require(getrusage(RUSAGE_SELF, &last) == 0, "getrusage");
  std::cout << seed << '\t' << layout << '\t' << cold << '\t' << ids.size()
            << '\t' << elapsed << '\t' << before << '\t' << d.resident(layout)
            << '\t' << last.ru_majflt - first.ru_majflt << '\t' << sum << '\n';
  return sum;
}
int main(int argc, char **argv) {
  if (argc != 2 || sysconf(_SC_PAGESIZE) != 4096)
    return 2;
  Data d(argv[1]);
  for (size_t i = 0; i < d.rows; ++i)
    require(d.read(i, 0) == d.read(i, 1) && d.read(i, 0) == d.read(i, 2),
            "row mismatch");
  std::cerr << "verified " << d.rows << " complete rows\n";
  for (int seed = 0; seed < 6; ++seed) {
    std::mt19937 rng(901 + seed);
    std::uniform_int_distribution<size_t> rand(0, d.rows - 1);
    std::vector<size_t> ids(4096);
    for (auto &i : ids)
      i = rand(rng);
    uint64_t control = 0;
    // Six seeds exercise all permutations of the three layouts.
    std::array<int, 3> order = {0, 1, 2};
    for (int j = 0; j < seed; ++j)
      std::next_permutation(order.begin(), order.end());
    for (int layout : order) {
      auto sum = bench(d, layout, true, seed, ids);
      require(sum == bench(d, layout, false, seed, ids), "warm checksum");
      if (!control)
        control = sum;
      else
        require(control == sum, "layout checksum");
    }
  }
}
