import ast
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pynvml as nv

nv.nvmlInit()
gpu = nv.nvmlDeviceGetHandleByUUID(os.environ["CUDA_VISIBLE_DEVICES"])
prompt = Path("fixtures.ids").read_text().splitlines()[1].split()
shared = dict(
    os.environ,
    LEANSONIC_Q38_MTP_TOKENS="160",
    LEANSONIC_Q38_MTP_PROMPT=",".join(prompt),
    LEANSONIC_MTP_REPLAY_IDS="1",
    LEANSONIC_Q38_MODEL=str(Path("direct-engram").resolve()),
    LEANSONIC_QWEN38_MTP_CUBIN_PATH=str(Path("exp4-mtp.cubin").resolve()),
)


def matches(log_text, pattern):
    found = re.search(pattern, log_text)
    if not found:
        raise ValueError(f"missing field: {pattern}")
    return found.groups()


records = []
for seq, arm in enumerate(("stock", "palette", "palette", "stock"), 1):
    loc = Path("exp4-mtp-probe") / arm / "release/deps"
    bins = [
        p for p in loc.glob("leansonic_core-*") if os.access(p, os.X_OK) and p.is_file()
    ]
    assert len(bins) == 1
    env = {
        **shared,
        "LEANSONIC_QWEN38_CUBIN_PATH": str(
            Path("wave1.cubin" if arm == "stock" else "exp4-word.cubin").resolve()
        ),
    }
    log = Path(f"exp4-mtp-r2-{seq}-{arm}.log")
    peak = 0
    with log.open("x") as f:
        process = subprocess.Popen(
            [
                str(bins[0]),
                "--ignored",
                "--exact",
                "qwen38_mtp::tests::mtp_greedy_acceptance",
                "--nocapture",
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            for item in nv.nvmlDeviceGetComputeRunningProcesses(gpu):
                if (
                    item.pid == process.pid
                    and 0 < item.usedGpuMemory <= nv.nvmlDeviceGetMemoryInfo(gpu).total
                ):
                    peak = max(peak, int(item.usedGpuMemory))
            time.sleep(0.025)
        if process.returncode:
            raise RuntimeError(log.read_text()[-1200:])
    s = log.read_text()
    hits, tries = map(int, matches(s, r"MTP greedy acceptance: (\d+)/(\d+)"))
    target, mstep = map(
        float,
        matches(s, r"per position: target forward ([\d.]+) ms, MTP step ([\d.]+) ms"),
    )
    (flag,) = matches(s, r"MTP MoE in VRAM: (true|false)")
    target_ids = ast.literal_eval(matches(s, r"target_ids=(\[[^\n]+\])")[0])
    draft_ids = ast.literal_eval(matches(s, r"draft_ids=(\[[^\n]+\])")[0])
    assert (
        target_ids[: len(prompt)] == list(map(int, prompt))
        and len(target_ids) == len(prompt) + 160
        and peak > 0
    )
    if records:
        assert (
            target_ids == records[0]["target_ids"]
            and draft_ids == records[0]["draft_ids"]
        )
    records.append(
        {
            "arm": arm,
            "moe_in_vram": flag == "true",
            "target_ms": target,
            "mtp_ms": mstep,
            "hits": hits,
            "tries": tries,
            "sampled_peak_bytes": peak,
            "target_ids": target_ids,
            "draft_ids": draft_ids,
        }
    )
    print(seq, arm, flag, target, mstep, hits, tries, peak, flush=True)
Path("exp4-mtp-results.json").write_text(
    json.dumps(
        {
            "prompt_ids": list(map(int, prompt)),
            "gpu": nv.nvmlDeviceGetName(gpu),
            "runs": records,
        },
        indent=2,
    )
    + "\n"
)
