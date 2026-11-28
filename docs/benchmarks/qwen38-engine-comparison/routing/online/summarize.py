"""Summarize numbered compare_stream launches and verify all greedy prefixes."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


def summarize(folder):
    outputs = {}
    prompts = {}
    groups = defaultdict(list)
    peaks = defaultdict(list)
    launches = []
    samples = []
    log_re = re.compile(r"prompt_rows=(\d+) moved=(\d+) bytes=(\d+) update_ms=([.\d]+)")
    for path in sorted(folder.glob("[0-9]*-*.json"), key=lambda p: int(p.stem.split("-")[0])):
        slot_limit = int(path.stem.split("-")[1])
        run = json.loads(path.read_text())
        stats = log_re.findall(path.with_suffix(".log").read_text())
        if slot_limit and len(stats) != len(run["records"]):
            raise ValueError(f"missing placement records for {path}")
        peaks[slot_limit].append(run["sampled_process_peak_bytes"])
        launches.append({"run": path.stem, "gpu": run["gpu"], "peak_bytes": run["sampled_process_peak_bytes"],
                         "launch_to_ready_ms": run["launch_to_ready_ms"]})
        for index, row in enumerate(run["records"]):
            case = row["case"] % 4
            prompt = row["prompt"]
            generated = row["generated"]
            n = len(generated)
            if prompts.setdefault(case, prompt) != prompt:
                raise ValueError("different input tokens")
            known = outputs.setdefault(case, generated)
            overlap = min(n, len(known))
            if known[:overlap] != generated[:overlap]:
                raise ValueError("different greedy output")
            if n > len(known):
                outputs[case] = generated
            if row["cycle"] == 0:
                continue
            sample = {k: row[k] for k in ("prefill_to_first_ms", "decode_ms", "last_to_done_ms", "total_to_done_ms")}
            sample["prefill_tok_s"] = 1000 * len(prompt) / sample["prefill_to_first_ms"]
            sample["decode_tok_s"] = 1000 * (n - 1) / sample["decode_ms"]
            if slot_limit:
                rows, moved, size, ms = stats[index]
                if int(rows) != len(prompt):
                    raise ValueError("wrong adaptation record")
                sample.update(moved=int(moved), transfer_bytes=int(size), update_ms=float(ms))
            groups[(n, case, slot_limit)].append(sample)
            samples.append(dict(run=path.stem, cycle=row["cycle"], case=case, output_tokens=n, **sample))
    if not launches:
        raise ValueError("no completed launches")
    summaries = []
    for (n, case, slots), rows in sorted(groups.items()):
        summaries.append(dict(case=case, output_tokens=n, slots=slots, samples=len(rows),
                              **{k: round(median(r[k] for r in rows), 4) for k in rows[0]}))
    return {"launches": launches, "peak_bytes_by_slot_limit": {k: max(v) for k, v in peaks.items()},
            "prompts": prompts, "greedy_prefix_verified_with_warmups": outputs, "medians": summaries,
            "teacher_parity_vs_stock_q4": json.loads((folder / "parity.json").read_text()), "samples": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.folder), indent=2))
