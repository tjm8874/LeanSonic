#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#define RT(x)                                                                  \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));             \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)
#define DR(x)                                                                  \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *s;                                                           \
      cuGetErrorString(e, &s);                                                 \
      std::fprintf(stderr, "%s: %s\n", #x, s);                                 \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)
template <class T> T *upload(const std::vector<T> &x) {
  T *p = nullptr;
  RT(cudaMalloc(&p, x.size() * sizeof(T)));
  RT(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
float fp32(uint16_t bf) {
  uint32_t u = uint32_t(bf) << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
struct Output {
  std::vector<int8_t> q;
  std::vector<float> scale;
};
int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  RT(cudaSetDevice(0));
  RT(cudaFree(nullptr));
  CUmodule m;
  DR(cuModuleLoad(&m, argv[1]));
  CUfunction funcs[2];
  DR(cuModuleGetFunction(&funcs[0], m, "ink2_gqa_rel_sink_attn_q8_decode"));
  DR(cuModuleGetFunction(&funcs[1], m, "ink2_gqa_attn_g1_q8_decode"));
  std::puts("seed,global,pos,path,code_diff_fp64,max_scale_abs,rmse_dequant_vs_"
            "fp64,pair_code_diff,pair_scale_bit_diff");
  for (unsigned seed : {107u, 303u, 419u}) {
    std::mt19937 rng(seed);
    auto random = [&]() { return (int(rng() % 4097) - 2048) / 2048.f; };
    int stride = 6552;
    std::vector<float> features(stride);
    for (float &x : features)
      x = random();
    std::vector<uint16_t> rel(16 * 1024);
    for (auto &z : rel) {
      float x = random() * .125f;
      uint32_t bits;
      std::memcpy(&bits, &x, 4);
      z = bits >> 16;
    }
    std::vector<__half> k(1536 * 1024), v(k.size());
    for (auto &x : k)
      x = __float2half(random());
    for (auto &x : v)
      x = __float2half(random());
    auto df = upload(features);
    auto dr = upload(rel);
    auto dk = upload(k);
    auto dv = upload(v);
    for (int global : {0, 1})
      for (int pos : {0, 1, 7, 31, 255, 256, 330, 511, 1023, 1024, 1535}) {
        int cap = global ? 1536 : 1024;
        int first = global ? 0 : std::max(0, pos + 1 - 1024);
        std::vector<double> gold(3072), oracle_sc(48);
        std::vector<int> oracle_q(3072);
        for (int h = 0; h < 24; h++) {
          const int kv = h / 3;
          std::vector<double> logits(pos + 1 - first);
          double highest = features[6528 + h];
          for (int j = first; j <= pos; j++) {
            const int r = j % cap;
            double dot = 0.;
            for (int i = 0; i < 128; i++)
              dot += features[h * 128 + i] *
                     double(__half2float(k[r * 1024 + kv * 128 + i]));
            double x = dot / std::sqrt(128.);
            int dist = pos - j;
            if (dist < 1024)
              for (int i = 0; i < 16; i++)
                x += features[3072 + h * 16 + i] *
                     double(fp32(rel[i * 1024 + dist]));
            logits[j - first] = x;
            highest = std::max(highest, x);
          }
          double den = std::exp(features[6528 + h] - highest);
          for (int j = first; j <= pos; j++) {
            double w = std::exp(logits[j - first] - highest);
            den += w;
            for (int i = 0; i < 128; i++)
              gold[h * 128 + i] +=
                  w * double(__half2float(v[(j % cap) * 1024 + kv * 128 + i]));
          }
          for (int i = 0; i < 128; i++)
            gold[h * 128 + i] /=
                den * (1. + std::exp(-double(features[3456 + h * 128 + i])));
        }
        for (int blk = 0; blk < 48; blk++) {
          double mx = 0.;
          for (int i = 0; i < 64; i++)
            mx = std::max(mx, std::abs(gold[blk * 64 + i]));
          oracle_sc[blk] = mx / 127.;
          for (int i = 0; i < 64; i++)
            oracle_q[blk * 64 + i] =
                mx ? int(std::nearbyint(gold[blk * 64 + i] * 127. / mx)) : 0;
        }
        auto dp = upload(std::vector<int>{pos});
        void *null = nullptr;
        Output out[2];
        for (int mode = 0; mode < 2; mode++) {
          auto dq = upload(std::vector<int8_t>(3072, -128));
          auto ds = upload(std::vector<float>(48, NAN));
          void *args[] = {&df,   &stride, &dr,  &dk,     &dv, &null,
                          &null, &dp,     &cap, &global, &dq, &ds};
          DR(cuLaunchKernel(funcs[mode], mode ? 8 : 24, 1, 1, 256, 1, 1, 0,
                            nullptr, args, nullptr));
          RT(cudaDeviceSynchronize());
          out[mode].q.resize(3072);
          out[mode].scale.resize(48);
          RT(cudaMemcpy(out[mode].q.data(), dq, 3072, cudaMemcpyDeviceToHost));
          RT(cudaMemcpy(out[mode].scale.data(), ds, 48 * 4,
                        cudaMemcpyDeviceToHost));
          RT(cudaFree(dq));
          RT(cudaFree(ds));
        }
        int changed = 0, sc_changed = 0;
        for (int i = 0; i < 3072; i++)
          changed += out[0].q[i] != out[1].q[i];
        for (int i = 0; i < 48; i++)
          sc_changed += std::memcmp(&out[0].scale[i], &out[1].scale[i], 4) != 0;
        for (int mode = 0; mode < 2; mode++) {
          int errors = 0;
          double max_scale = 0., sse = 0.;
          for (int i = 0; i < 3072; i++) {
            auto scale = out[mode].scale[i / 64];
            if (!std::isfinite(scale) || scale < 0 || out[mode].q[i] == -128)
              return 4;
            errors += out[mode].q[i] != oracle_q[i];
            max_scale =
                std::max(max_scale, std::abs(scale - oracle_sc[i / 64]));
            double err = out[mode].q[i] * double(scale) - gold[i];
            sse += err * err;
          }
          std::printf("%u,%d,%d,%s,%d,%.12g,%.12g,%d,%d\n", seed, global, pos,
                      mode ? "g1" : "fallback", errors, max_scale,
                      std::sqrt(sse / 3072), changed, sc_changed);
        }
        RT(cudaFree(dp));
      }
    RT(cudaFree(df));
    RT(cudaFree(dr));
    RT(cudaFree(dk));
    RT(cudaFree(dv));
  }
  DR(cuModuleUnload(m));
}
