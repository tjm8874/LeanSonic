"""Run fixed greedy requests after evicting only private Engram pages. Linux, 4 KiB pages."""

import argparse
import ctypes
import json
import mmap
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path


def ranges(model):
    mapping = json.loads((model / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    selected = {name for k, name in mapping.items() if ".ngram_embedding.shard_" in k}
    result = {}
    for name in selected:
        with (model / name).open("rb") as f:
            (header,) = struct.unpack("<Q", f.read(8))
            h = json.loads(f.read(header))
        spans = []
        for key, t in h.items():
            if ".ngram_embedding.shard_" not in key:
                continue
            lo, hi = t["data_offsets"]
            start = (lo + header + 8 + 4095) // 4096 * 4096
            end = (hi + header + 8) // 4096 * 4096
            if end > start:
                spans.append((start, end))
        result[name] = spans
    return result


def private_copy(model, out):
    selected = ranges(model)
    out.mkdir()
    for file in model.iterdir():
        if file.name in selected:
            shutil.copyfile(file, out / file.name)
        else:
            os.link(file, out / file.name)


def residency(model, evict):
    cached, total = 0, 0
    libc = ctypes.CDLL(None, use_errno=True)
    for name, spans in ranges(model).items():
        path = model / name
        if path.stat().st_nlink != 1:
            raise ValueError(f"shared file: {path}")
        with (
            path.open("rb") as f,
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as mapping,
        ):
            base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            for lo, hi in spans:
                if evict:
                    os.posix_fadvise(f.fileno(), lo, hi - lo, os.POSIX_FADV_DONTNEED)
                count = (hi - lo) // 4096
                buf = ctypes.create_string_buffer(count)
                if libc.mincore(
                    ctypes.c_void_p(base + lo), ctypes.c_size_t(hi - lo), buf
                ):
                    raise OSError(ctypes.get_errno(), "mincore")
                cached += buf.raw.count(b"\x01")
                total += count
    return cached, total


def trial(binary, model, prompt, log):
    residency(model, True)
    with log.open("x") as stderr:
        proc = subprocess.Popen(
            [str(binary), str(model), "256"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        try:

            def line():
                s = proc.stdout.readline()
                if not s:
                    raise RuntimeError(f"unexpected engine exit {proc.poll()}")
                return s.strip()

            while not line().startswith("READY"):
                pass
            prefetched = residency(model, False)
            before = residency(model, True)
            output = []
            for repeat in range(2):
                major0 = int(
                    Path(f"/proc/{proc.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[9]
                )
                start = time.monotonic_ns()
                proc.stdin.write("GEN 128 " + ",".join(map(str, prompt)) + "\n")
                proc.stdin.flush()
                tokens = []
                stamps = []
                while True:
                    msg = line()
                    if msg.startswith("T "):
                        tokens.append(int(msg[2:]))
                        stamps.append(time.monotonic_ns())
                    elif msg.startswith("DONE"):
                        break
                if len(tokens) != 128:
                    raise ValueError("incomplete request")
                major1 = int(
                    Path(f"/proc/{proc.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[9]
                )
                output.append(
                    dict(
                        prefill_ms=(stamps[0] - start) / 1e6,
                        decode_ms=(stamps[-1] - stamps[0]) / 1e6,
                        major_faults=major1 - major0,
                        tokens=tokens,
                    )
                )
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            if proc.wait(timeout=30):
                raise RuntimeError("unclean quit")
            return dict(
                resident_after_ready=prefetched,
                resident_before_first=before,
                requests=output,
            )
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("copy")
    prep.add_argument("model", type=Path)
    prep.add_argument("out", type=Path)
    run = sub.add_parser("run")
    for k in ("binary", "original", "tile", "prompts", "out"):
        run.add_argument(k, type=Path)
    a = p.parse_args()
    if a.mode == "copy":
        private_copy(a.model, a.out)
        return
    prompts = [list(map(int, l.split())) for l in a.prompts.read_text().splitlines()]
    a.out.mkdir()
    for i, case in enumerate((1, 4, 8, 12)):
        order = ("original", "tile") if i % 2 == 0 else ("tile", "original")
        outputs = []
        for arm in order:
            result = trial(
                a.binary, getattr(a, arm), prompts[case], a.out / f"{case}-{arm}.log"
            )
            result.update(case=case, arm=arm, prompt=prompts[case])
            (a.out / f"{case}-{arm}.json").write_text(json.dumps(result) + "\n")
            outputs.extend(r["tokens"] for r in result["requests"])
        if any(ids != outputs[0] for ids in outputs):
            raise ValueError("cross-layout output mismatch")
    (a.out / "complete").touch()


if __name__ == "__main__":
    main()
