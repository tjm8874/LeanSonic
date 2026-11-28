"""Compare a batch of HC down/injection rows against independent original scalar launches."""

import argparse
import ctypes as c
from pathlib import Path

import numpy as np


def run(ref, trial, require_exact):
    lib = c.CDLL("libcuda.so.1")

    def call(name, *params):
        result = getattr(lib, name)(*params)
        if result:
            raise RuntimeError(f"{name}: CUDA status {result}")

    call("cuInit", 0)
    ctx = c.c_void_p()
    call("cuDevicePrimaryCtxRetain", c.byref(ctx), 0)
    call("cuCtxSetCurrent", ctx)
    allocated = []

    def upload(x):
        ptr = c.c_uint64()
        call("cuMemAlloc_v2", c.byref(ptr), c.c_size_t(x.nbytes))
        call("cuMemcpyHtoD_v2", ptr, c.c_void_p(x.ctypes.data), c.c_size_t(x.nbytes))
        allocated.append(ptr)
        return ptr

    def read(ptr, shape):
        x = np.empty(shape, np.float32)
        call("cuMemcpyDtoH_v2", c.c_void_p(x.ctypes.data), ptr, c.c_size_t(x.nbytes))
        return x

    def func(mod, name):
        f = c.c_void_p()
        call("cuModuleGetFunction", c.byref(f), mod, name.encode())
        return f

    def launch(f, grid, args):
        pointers = (c.c_void_p * len(args))(*(c.addressof(x) for x in args))
        call(
            "cuLaunchKernel", f, grid[0], grid[1], 1, 256, 1, 1, 0, None, pointers, None
        )

    def offset(ptr, n):
        return c.c_uint64(ptr.value + n)

    base, mod = c.c_void_p(), c.c_void_p()
    call("cuModuleLoad", c.byref(base), str(ref).encode())
    call("cuModuleLoad", c.byref(mod), str(trial).encode())
    scalar = func(base, "q38_hc_down_inj_10k")
    batched = func(mod, "q38_hc_down_inj_10k_b4")
    rng = np.random.default_rng(227)

    def params(n):
        codes = rng.integers(0, 1 << 32, size=(n, 320, 4), dtype=np.uint32)

        def bf16():
            return (
                rng.normal(0, 0.1, (n, 320)).astype(np.float32).view(np.uint32) >> 16
            ).astype(np.uint16)

        return [upload(codes), upload(bf16()), upload(bf16())]

    down = params(320)
    inject = params(4)
    max_batch = 64
    x = upload(rng.normal(0, 1, (max_batch, 10240)).astype(np.float32))
    low, out, other_low, other_out = [
        upload(np.zeros((max_batch, n), np.float32)) for n in [320, 4, 320, 4]
    ]
    exact = True
    for injection in (0, 1):
        enabled = c.c_int(injection)
        inp = inject if injection else [c.c_uint64(0) for _ in range(3)]
        for n in (1, 2, 3, 4, 5, 8, 16, 64):
            for i in range(n):
                launch(
                    scalar,
                    (80 + injection, 1),
                    down
                    + inp
                    + [
                        offset(x, i * 10240 * 4),
                        offset(low, i * 320 * 4),
                        offset(out, i * 4 * 4),
                        enabled,
                    ],
                )
            launch(
                batched,
                (40 + injection, (n + 3) // 4),
                down + inp + [x, other_low, other_out, c.c_int(n), enabled],
            )
            for name, a, b, cols in [
                ("low", low, other_low, 320),
                ("inject", out, other_out, 4),
            ]:
                u, v = (read(ptr, (n, cols)) for ptr in (a, b))
                if not (np.isfinite(u).all() and np.isfinite(v).all()):
                    raise ValueError("non-finite output")
                errors = int(np.count_nonzero(u.view(np.uint32) != v.view(np.uint32)))
                exact &= errors == 0
                print(
                    n,
                    injection,
                    name,
                    u.size,
                    errors,
                    float(np.max(np.abs(u - v))),
                    sep="\t",
                )
    for ptr in allocated:
        call("cuMemFree_v2", ptr)
    call("cuModuleUnload", mod)
    call("cuModuleUnload", base)
    call("cuDevicePrimaryCtxRelease_v2", 0)
    if require_exact and not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--require-exact", action="store_true")
    a = p.parse_args()
    run(a.reference, a.candidate, a.require_exact)
