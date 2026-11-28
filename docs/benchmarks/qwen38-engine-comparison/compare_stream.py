"""Compare resident greedy runs over identical token IDs, timing the streaming boundary."""

import argparse
import functools
import json
import operator
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pynvml as nv

now = time.monotonic_ns


def deadline(signum, frame):
    raise TimeoutError("engine benchmark timed out")


def run(args):
    nv.nvmlInit()
    gpu = nv.nvmlDeviceGetHandleByUUID(os.environ["CUDA_VISIBLE_DEVICES"])
    fixtures = [
        list(map(int, line.split())) for line in args.fixtures.read_text().splitlines()
    ]
    whole = functools.reduce(operator.iadd, fixtures, [])
    prompts = [
        fixtures[1],
        (whole * max(1, (128 + len(whole) - 1) // len(whole)))[:128],
    ]
    if len(prompts[1]) != 128:
        raise ValueError("long fixture length")
    records = []
    peak = [0]
    event = threading.Event()
    with args.log.open("x") as log:
        start = now()
        proc = subprocess.Popen(
            args.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )

        def sample():
            for row in nv.nvmlDeviceGetComputeRunningProcesses(gpu):
                if (
                    row.pid == proc.pid
                    and isinstance(row.usedGpuMemory, int)
                    and 0 < row.usedGpuMemory <= nv.nvmlDeviceGetMemoryInfo(gpu).total
                ):
                    peak[0] = max(peak[0], int(row.usedGpuMemory))

        def monitor():
            while not event.is_set():
                sample()
                event.wait(0.025)

        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()

        def readline():
            s = proc.stdout.readline()
            if not s:
                raise RuntimeError(f"engine exited with {proc.poll()}")
            s = s.strip()
            if s.startswith("ERR "):
                raise RuntimeError(s)
            return s

        def send(line):
            proc.stdin.write(line + "\n")
            proc.stdin.flush()

        previous_handler = signal.signal(signal.SIGALRM, deadline)
        try:
            signal.alarm(1200)
            while not readline().startswith("READY"):
                pass
            ready = now()
            for cycle in range(4):
                for case, ids in enumerate(prompts):
                    n = 32
                    sendtime = now()
                    send(f"GEN {n} " + ",".join(map(str, ids)))
                    stamps = []
                    tokens = []
                    done = None
                    while True:
                        line = readline()
                        ts = now()
                        if line.startswith("RESUME ") and int(line.split()[1]) > 0:
                            raise ValueError("prompt cache reuse")
                        if line.startswith("T "):
                            tokens.append(int(line.split()[1]))
                            stamps.append(ts)
                        if line.startswith("DONE"):
                            done = ts
                            fields = line.split()
                            if len(fields) > 8 and int(fields[8]) != 0:
                                raise ValueError("engine reported reused prompt tokens")
                            break
                    if len(tokens) != n:
                        raise ValueError(f"expected {n}, got {len(tokens)}")
                    records.append(
                        {
                            "cycle": cycle,
                            "case": case,
                            "prompt": ids,
                            "generated": tokens,
                            "prefill_to_first_ms": (stamps[0] - sendtime) / 1e6,
                            "decode_ms": (stamps[-1] - stamps[0]) / 1e6,
                            "last_to_done_ms": (done - stamps[-1]) / 1e6,
                            "total_to_done_ms": (done - sendtime) / 1e6,
                        }
                    )
                    print(cycle, case, round((done - sendtime) / 1e9, 3), flush=True)
            sample()
            send("QUIT")
            proc.wait(timeout=90)
            if proc.returncode:
                raise RuntimeError(f"engine exit {proc.returncode}")
            if peak[0] <= 0:
                raise ValueError("no per-process VRAM samples")
        finally:
            event.set()
            worker.join()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    distinct = [
        len({tuple(r["generated"]) for r in records if r["case"] == i})
        for i in range(len(prompts))
    ]
    return {
        "distinct_sequences_including_warmup": distinct,
        "gpu": nv.nvmlDeviceGetName(gpu),
        "sampled_process_peak_bytes": peak[0],
        "launch_to_ready_ms": (ready - start) / 1e6,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    result = run(args)
    with args.result.open("x") as out:
        json.dump(result, out, indent=2)
