// Diagnostic for one complete BF16 matrix and its already packed native copy.
// Verify the existing Q8 codes and scales before writing the candidate.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static float bf16(unsigned word) { return std::bit_cast<float>(word << 16); }

int main(int argc, char** argv) {
    if (argc != 7) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    std::fstream output(argv[4], std::ios::binary | std::ios::in | std::ios::out);
    const size_t blocks = std::stoull(argv[3]);
    const size_t code_offset = std::stoull(argv[5]);
    const size_t scale_offset = std::stoull(argv[6]);
    std::vector<uint16_t> source(blocks * 64);
    std::vector<int8_t> codes(blocks * 64);
    std::vector<float> scales(blocks);
    input.seekg(std::stoull(argv[2]));
    input.read(reinterpret_cast<char*>(source.data()), source.size() * 2);
    output.seekg(code_offset);
    output.read(reinterpret_cast<char*>(codes.data()), codes.size());
    output.seekg(scale_offset);
    output.read(reinterpret_cast<char*>(scales.data()), scales.size() * 4);
    if (!input || !output) throw std::runtime_error("matrix read failed");

    long invalid = 0, changed = 0;
    double original_sse = 0, candidate_sse = 0;
    #pragma omp parallel for reduction(+:invalid,changed,original_sse,candidate_sse)
    for (size_t block = 0; block < blocks; ++block) {
        const auto* data = source.data() + 64 * block;
        auto* dst = codes.data() + 64 * block;
        unsigned maximum = 0;
        for (int j = 0; j < 64; ++j) {
            maximum = std::max(maximum, unsigned(data[j] & 0x7fff));
        }
        if (maximum >= 0x7f80) {
            ++invalid;
            continue;
        }
        float best_scale = 0;
        double best_error = 0;
        std::array<int8_t, 64> best_codes{};
        for (int delta : {0, -1, 1, -2, 2}) {
            int word = int(maximum) + delta;
            if (word < 0 || word >= 0x7f80) continue;
            const float high = bf16(word);
            const float scale = high * (1.0f / 127.f);
            const float inv = high > 0 ? 127.f / high : 0.f;
            std::array<int8_t, 64> trial{};
            double error = 0;
            for (int j = 0; j < 64; ++j) {
                const float x = bf16(data[j]);
                trial[j] = static_cast<int8_t>(std::clamp(std::round(x * inv), -127.f, 127.f));
                const double diff = double(x) - double(trial[j]) * double(scale);
                error += diff * diff;
            }
            if (delta == 0) {
                if (scales[block] != scale || !std::equal(trial.begin(), trial.end(), dst)) ++invalid;
                original_sse += error;
            }
            if (delta == 0 || error < best_error) {
                best_error = error;
                best_scale = scale;
                best_codes = trial;
            }
        }
        candidate_sse += best_error;
        if (scales[block] != best_scale) {
            ++changed;
            scales[block] = best_scale;
            std::copy(best_codes.begin(), best_codes.end(), dst);
        }
    }
    if (invalid) {
        std::fprintf(stderr, "%ld baseline blocks differ\n", invalid);
        return 1;
    }
    output.seekp(code_offset);
    output.write(reinterpret_cast<const char*>(codes.data()), codes.size());
    output.seekp(scale_offset);
    output.write(reinterpret_cast<const char*>(scales.data()), scales.size() * 4);
    output.close();
    if (!output) throw std::runtime_error("matrix write failed");
    std::printf("%zu %ld %.17g %.17g\n", blocks, changed, original_sse, candidate_sse);
}
