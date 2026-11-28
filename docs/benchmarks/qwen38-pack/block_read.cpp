// Linux microbenchmark for warm, direct-to-buffer expert weight reads.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <zstd.h>

constexpr size_t block = 256 * 1024;
struct Entry {
  uint64_t start;
  size_t size;
  bool raw;
};

void read_at(int fd, uint64_t offset, uint8_t *out, size_t size) {
  while (size) {
    ssize_t n = pread(fd, out, size, offset);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("failed or incomplete pread");
    size -= n;
    out += n;
    offset += n;
  }
}

void run(int raw_fd, uint64_t base, int packed_fd,
         const std::vector<Entry> &entries, std::vector<uint8_t> &dest,
         bool packed, size_t workers) {
  const size_t count = entries.size();
  workers = std::min(workers, count);
  std::vector<std::thread> pool;
  std::vector<std::exception_ptr> errors(workers);
  for (size_t w = 0; w < workers; ++w) {
    pool.emplace_back([&, w] {
      try {
        size_t lo = w * count / workers, hi = (w + 1) * count / workers;
        if (!packed) {
          read_at(raw_fd, base + lo * block, dest.data() + lo * block,
                  (hi - lo) * block);
          return;
        }
        std::vector<uint8_t> scratch(block);
        for (size_t i = lo; i < hi; ++i) {
          auto e = entries.at(i);
          uint8_t *out = dest.data() + i * block;
          read_at(packed_fd, e.start, e.raw ? out : scratch.data(), e.size);
          if (!e.raw &&
              ZSTD_decompress(out, block, scratch.data(), e.size) != block)
            throw std::runtime_error("invalid zstd block");
        }
      } catch (...) {
        errors[w] = std::current_exception();
      }
    });
  }
  for (auto &thread : pool)
    thread.join();
  for (auto error : errors)
    if (error)
      std::rethrow_exception(error);
}

int main(int argc, char **argv) try {
  if (argc != 5)
    throw std::runtime_error("expected ORIGINAL OFFSET BYTES PACKED");
  size_t base = std::stoull(argv[2]), bytes = std::stoull(argv[3]);
  if (!bytes || bytes % block ||
      base > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      bytes > std::numeric_limits<int64_t>::max() - base)
    throw std::runtime_error("expected whole 256 KiB blocks");
  int raw_fd = open(argv[1], O_RDONLY), packed_fd = open(argv[4], O_RDONLY);
  if (raw_fd < 0 || packed_fd < 0)
    throw std::runtime_error("cannot open input");
  size_t count = bytes / block;
  std::vector<uint8_t> index(8 * count);
  read_at(packed_fd, 0, index.data(), index.size());
  std::vector<Entry> entries;
  uint64_t offset = index.size();
  for (size_t i = 0; i < count; ++i) {
    uint64_t value = 0;
    for (int j = 7; j >= 0; --j)
      value = (value << 8) | index[8 * i + j];
    bool raw = (value >> 63) != 0;
    size_t length = value & ~(uint64_t(1) << 63);
    if (!length || length > block || (raw && length != block))
      throw std::runtime_error("invalid block length");
    entries.push_back({offset, length, raw});
    offset += length;
  }
  if (lseek(packed_fd, 0, SEEK_END) != static_cast<off_t>(offset))
    throw std::runtime_error("invalid packed length");
  std::vector<uint8_t> reference(bytes), dest(bytes);
  read_at(raw_fd, base, reference.data(), bytes);
  for (size_t workers : {1, 2, 4, 8}) {
    for (int round = 0; round < 5; ++round) {
      for (int arm = 0; arm < 2; ++arm) {
        bool packed = (round + arm) % 2;
        std::fill(dest.begin(), dest.end(), 0);
        auto start = std::chrono::steady_clock::now();
        run(raw_fd, base, packed_fd, entries, dest, packed, workers);
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        if (dest != reference)
          throw std::runtime_error("output mismatch");
        if (round)
          std::cout << workers << ',' << packed << ',' << ns << '\n';
      }
    }
  }
  close(raw_fd);
  close(packed_fd);
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
