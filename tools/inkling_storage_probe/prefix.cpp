// Research codec for Inkling NVFP4 expert codes. Four lanes each traverse an
// eight-byte block. The prior absolute maximum is known at decode time.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Byte = uint8_t;
using Word = uint32_t;
constexpr Word total = 4096, lower = 1u << 23;
constexpr size_t tile_size = 128 * 1024;
constexpr size_t contexts = 8 * 8;
using Counts = std::array<uint32_t, 256>;
using Table = std::array<uint16_t, 256>;
using Tables = std::array<Table, contexts>;

static void require(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
static Byte next_max(Byte previous, Byte v) {
  return std::max({previous, Byte(v & 7), Byte((v >> 4) & 7)});
}
static Table normalize(const Counts &hist) {
  int64_t n = 0;
  for (auto v : hist)
    n += v;
  Table f{};
  if (n == 0) {
    f[0] = total;
    return f;
  }
  std::array<int64_t, 256> err{};
  int sum = 0;
  for (int i = 0; i < 256; ++i) {
    if (hist[i] == 0)
      continue;
    int64_t ideal = int64_t(hist[i]) * total;
    f[i] = std::max<int64_t>(1, ideal / n);
    err[i] = ideal - f[i] * n;
    sum += f[i];
  }
  while (sum != int(total)) {
    int best = -1;
    bool up = sum < int(total);
    for (int i = 0; i < 256; ++i) {
      if ((up && f[i] > 0) || (!up && f[i] > 1)) {
        if (best < 0 || (up ? err[i] > err[best] : err[i] < err[best]))
          best = i;
      }
    }
    require(best >= 0, "frequency normalization");
    f[best] += up ? 1 : -1;
    err[best] += up ? -n : n;
    sum += up ? 1 : -1;
  }
  return f;
}
static void put(std::ostream &out, Word v, int width) {
  for (int i = 0; i < width; ++i)
    out.put(char(v >> (8 * i)));
}
static Word get(std::istream &in, int width) {
  Word v = 0;
  for (int i = 0; i < width; ++i) {
    int b = in.get();
    require(b >= 0, "short metadata");
    v |= Word(b) << (8 * i);
  }
  return v;
}
// One symbol is an unsigned byte or, in nibble mode, an E2M1 code.
template <bool N> struct Codec {
  static constexpr int width = N ? 16 : 8, alphabet = N ? 16 : 256,
                       count = width * 8 * 4;
  struct History {
    unsigned largest = 0, sum = 0;
    unsigned context(int j) const {
      const unsigned n = unsigned(j) * (N ? 1 : 2);
      const unsigned group = (sum > 2 * n) + (sum > 3 * n) + (sum > 4 * n);
      return (unsigned(j) * 8 + largest) * 4 + group;
    }
    void update(Byte v) {
      largest = next_max(largest, v);
      sum += (v & 7) + ((v >> 4) & 7);
    }
  };
  static Byte sym(const Byte *b, int j) {
    if constexpr (N)
      return (b[j / 2] >> (4 * (j % 2))) & 15;
    return b[j];
  }
  static void run(const std::vector<Byte> &raw, const char *dest) {
    std::vector<Counts> h(count);
    for (size_t off = 0; off < raw.size(); off += 8) {
      History state;
      for (int j = 0; j < width; ++j) {
        auto v = sym(raw.data() + off, j);
        ++h[state.context(j)][v];
        state.update(v);
      }
    }
    std::ofstream out(dest, std::ios::binary);
    std::vector<Table> freq(count), start(count);
    for (int c = 0; c < count; ++c) {
      freq[c] = normalize(h[c]);
      Word sum = 0;
      for (int i = 0; i < alphabet; ++i) {
        put(out, freq[c][i], 2);
        start[c][i] = sum;
        sum += freq[c][i];
      }
      require(sum == total, "bad frequencies");
    }
    for (size_t first = 0; first < raw.size(); first += tile_size) {
      std::array<Word, 4> states{lower, lower, lower, lower};
      std::vector<Byte> payload;
      for (size_t end = std::min(first + tile_size, raw.size()); end > first;
           end -= 32) {
        const Byte *block = raw.data() + end - 32;
        std::array<std::array<unsigned, width>, 4> ctx{};
        for (int lane = 0; lane < 4; ++lane) {
          History state;
          for (int j = 0; j < width; ++j) {
            ctx[lane][j] = state.context(j);
            state.update(sym(block + 8 * lane, j));
          }
        }
        for (int j = width - 1; j >= 0; --j)
          for (int lane = 3; lane >= 0; --lane) {
            auto c = ctx[lane][j];
            auto v = sym(block + lane * 8, j);
            Word f = freq[c][v], &x = states[lane];
            while (x >= ((lower >> 12) << 8) * f) {
              payload.push_back(Byte(x));
              x >>= 8;
            }
            x = (x / f << 12) + x % f + start[c][v];
          }
      }
      put(out, 16 + payload.size(), 4);
      for (auto x : states)
        put(out, x, 4);
      std::reverse(payload.begin(), payload.end());
      out.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    }
    while (size_t(out.tellp()) % 8)
      out.put(0);
    const auto size = out.tellp();
    out.close();
    require(bool(out), "write");
    auto t = std::chrono::steady_clock::now();
    std::ifstream in(dest, std::ios::binary);
    std::vector<std::array<Word, total>> table(count);
    for (auto &entry : table) {
      Word base = 0;
      for (Word v = 0; v < alphabet; ++v) {
        auto f = get(in, 2);
        require(base + f <= total, "invalid distribution");
        for (Word i = 0; i < f; ++i)
          entry[base + i] = (v << 24) | (i << 12) | (f - 1);
        base += f;
      }
      require(base == total, "incomplete distribution");
    }
    for (size_t first = 0; first < raw.size(); first += tile_size) {
      auto len = std::min(tile_size, raw.size() - first);
      auto frame = get(in, 4);
      require(frame >= 16 && frame <= 16 + len * 3, "frame length");
      std::array<Word, 4> states;
      for (auto &x : states) {
        x = get(in, 4);
        require(x >= lower && x < 256u * lower, "state");
      }
      std::vector<Byte> payload(frame - 16), decoded(len);
      in.read(reinterpret_cast<char *>(payload.data()), payload.size());
      require(bool(in), "payload");
      size_t cur = 0;
      for (size_t pos = 0; pos < len; pos += 32) {
        std::array<History, 4> history{};
        for (int j = 0; j < width; ++j)
          for (int lane = 0; lane < 4; ++lane) {
            auto &x = states[lane];
            auto &state = history[lane];
            auto entry = table[state.context(j)][x & 4095];
            Byte v = entry >> 24;
            state.update(v);
            if constexpr (N)
              decoded[pos + 8 * lane + j / 2] |= v << (4 * (j % 2));
            else
              decoded[pos + 8 * lane + j] = v;
            x = ((entry & 4095) + 1) * (x >> 12) + ((entry >> 12) & 4095);
            while (x < lower) {
              require(cur < payload.size(), "truncated");
              x = (x << 8) | payload[cur++];
            }
          }
      }
      require(cur == payload.size(), "trailing");
      for (auto x : states)
        require(x == lower, "end state");
      require(std::equal(decoded.begin(), decoded.end(), raw.begin() + first),
              "roundtrip");
    }
    while (size_t(in.tellg()) % 8)
      require(in.get() == 0, "padding");
    require(in.peek() < 0, "extra bytes");
    std::cout << size << " "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t)
                     .count()
              << "\n";
  }
};
int main(int argc, char **argv) {
  try {
    require(argc == 6,
            "usage: nvfp4_ctx shard offset length output byte|nibble");
    std::ifstream input(argv[1], std::ios::binary);
    input.seekg(std::stoull(argv[2]));
    std::vector<Byte> raw(std::stoull(argv[3]));
    require(!raw.empty() && raw.size() % 32 == 0 && raw.size() <= UINT32_MAX,
            "length");
    input.read(reinterpret_cast<char *>(raw.data()), raw.size());
    require(bool(input), "short raw data");
    std::uint64_t changed = 0;
    for (auto &value : raw) {
      Byte lo = value & 15, hi = value >> 4;
      changed += (lo == 8) + (hi == 8);
      value = Byte((lo == 8 ? 0 : lo) | ((hi == 8 ? 0 : hi) << 4));
    }
    std::cerr << changed << "\n";
    if (std::string(argv[5]) == "byte")
      Codec<false>::run(raw, argv[4]);
    else if (std::string(argv[5]) == "nibble")
      Codec<true>::run(raw, argv[4]);
    else
      throw std::runtime_error("mode");
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
