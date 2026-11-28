// Compare one-row, sequential-batched, and checkpointed GDN recurrence on
// identical inputs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
void check(CUresult r) {
  if (r != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(msg ? msg : "CUDA error");
  }
}
using F = std::vector<float>;
struct Dev {
  CUdeviceptr p;
  size_t n;
  explicit Dev(const F &x) : n(x.size()) {
    check(cuMemAlloc(&p, n * 4));
    check(cuMemcpyHtoD(p, x.data(), n * 4));
  }
  F read() {
    F x(n);
    check(cuMemcpyDtoH(x.data(), p, n * 4));
    return x;
  }
  ~Dev() { cuMemFree(p); }
  Dev(const Dev &) = delete;
};
bool diff(const char *field, int batch, int gate, const char *path, const F &a,
          const F &b) {
  if (a.size() != b.size())
    throw std::runtime_error("shape mismatch");
  size_t count = 0;
  float delta = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error("nonfinite");
    count += std::memcmp(&a[i], &b[i], 4) != 0;
    delta = std::max(delta, std::abs(a[i] - b[i]));
  }
  std::cout << batch << "\t" << gate << "\t" << path << "\t" << field << "\t"
            << a.size() << "\t" << count << "\t" << delta << "\n";
  return count == 0;
}
bool run(CUmodule reference_mod, CUmodule mod, int batch, int gate) {
  bool exact = true;
  std::mt19937 rand(9027);
  std::uniform_real_distribution<float> uniform(-1.5, 1.5);
  auto data = [&](size_t n) {
    F a(n);
    for (float &x : a)
      x = uniform(rand);
    return a;
  };
  constexpr int state_n = 48 * 128 * 128;
  F initial = data(state_n);
  Dev q(data(10240 * batch)), z(data(16480 * batch)), al(data(48)),
      dt(data(48)), norm(data(128));
  Dev state(initial), out(F(batch * 6144));
  auto launch = [&](CUmodule module, const char *name, void **args) {
    CUfunction f;
    check(cuModuleGetFunction(&f, module, name));
    check(cuLaunchKernel(f, 48, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
  };
  F checkpoints;
  for (int i = 0; i < batch; ++i) {
    CUdeviceptr qp = q.p + i * 10240 * 4, zp = z.p + (i * 16480 + 10240) * 4,
                op = out.p + i * 6144 * 4;
    void *args[] = {&qp, &zp, &al.p, &dt.p, &norm.p, &gate, &state.p, &op};
    launch(reference_mod, "q38_gdn_recurrent_step", args);
    if (i + 1 < batch) {
      F next = state.read();
      checkpoints.insert(checkpoints.end(), next.begin(), next.end());
    }
  }
  const F reference = state.read(), reference_out = out.read();
  {
    Dev control(initial), result(F(batch * 6144));
    for (int i = 0; i < batch; ++i) {
      CUdeviceptr qp = q.p + i * 10240 * 4, zp = z.p + (i * 16480 + 10240) * 4,
                  op = result.p + i * 6144 * 4;
      void *args[] = {&qp, &zp, &al.p, &dt.p, &norm.p, &gate, &control.p, &op};
      launch(mod, "q38_gdn_recurrent_step", args);
    }
    exact &= diff("state", batch, gate, "scalar", reference, control.read());
    exact &=
        diff("output", batch, gate, "scalar", reference_out, result.read());
  }
  for (bool checkpoint : {false, true}) {
    Dev candidate(initial), result(F(batch * 6144)),
        ck(F(std::max(batch - 1, 1) * state_n));
    CUdeviceptr cp = batch > 1 ? ck.p : 0;
    void *seq[] = {&q.p,  &z.p,   &al.p,        &dt.p,    &norm.p,
                   &gate, &batch, &candidate.p, &result.p};
    void *with_ck[] = {&q.p,  &z.p,   &al.p,        &dt.p, &norm.p,
                       &gate, &batch, &candidate.p, &cp,   &result.p};
    launch(mod,
           checkpoint ? "q38_gdn_recurrent_step_ckpt_b8"
                      : "q38_gdn_recurrent_step_seq_b",
           checkpoint ? with_ck : seq);
    const char *label = checkpoint ? "checkpoint" : "seq";
    exact &= diff("state", batch, gate, label, reference, candidate.read());
    exact &= diff("output", batch, gate, label, reference_out, result.read());
    if (checkpoint && batch > 1)
      exact &= diff("intermediate", batch, gate, label, checkpoints, ck.read());
  }
  return exact;
}
int main(int argc, char **argv) {
  const bool require_exact =
      argc == 4 && std::string(argv[3]) == "--require-exact";
  if (argc != 3 && !require_exact) {
    std::cerr << "usage: gdn_parity CONTROL.cubin CANDIDATE.cubin "
                 "[--require-exact]\n";
    return 2;
  }
  bool exact = true;
  check(cuInit(0));
  CUdevice dev;
  CUcontext ctx;
  CUmodule base, m;
  check(cuDeviceGet(&dev, 0));
  check(cuDevicePrimaryCtxRetain(&ctx, dev));
  check(cuCtxSetCurrent(ctx));
  check(cuModuleLoad(&base, argv[1]));
  check(cuModuleLoad(&m, argv[2]));
  for (int b = 1; b <= 8; ++b)
    for (int g : {0, 1})
      exact &= run(base, m, b, g);
  check(cuModuleUnload(base));
  check(cuModuleUnload(m));
  check(cuDevicePrimaryCtxRelease(dev));
  return require_exact && !exact ? 1 : 0;
}
