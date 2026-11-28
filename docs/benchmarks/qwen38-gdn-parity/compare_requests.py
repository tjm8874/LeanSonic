"""Summarize full 128-token comparisons, stopping-at-EOS parity and reproducibility."""
import argparse
import json
from pathlib import Path


def read(path):
    requests = {}
    for line in path.read_text().splitlines():
        cycle, case, depth, _, _, _, _, output = line.split("\t")
        key = (int(cycle), int(case), int(depth))
        tokens = json.loads(output)
        if key in requests or len(tokens) != 128:
            raise ValueError(f"{path}: duplicate or incomplete request {key}")
        requests[key] = tokens
    if not requests:
        raise ValueError(f"{path}: empty result")
    return requests


def first_difference(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)


def stopped(tokens, stops):
    for i, token in enumerate(tokens):
        if token in stops:
            return tokens[:i + 1]
    return tokens


def compare(baseline_path, candidate_path, stops):
    baseline, candidate = read(baseline_path), read(candidate_path)
    cycles = max(key[0] for key in baseline) + 1
    expected = {(cycle, case, depth) for cycle in range(cycles)
                for case in range(4) for depth in (0, 1, 2)}
    if baseline.keys() != expected or candidate.keys() != expected:
        raise ValueError("expected all four prompts and three depths for each cycle and arm")
    rows = []
    for case in range(4):
        for depth in (0, 1, 2):
            for cycle in range(cycles):
                key = (cycle, case, depth)
                row = {"case": case, "depth": depth, "cycle": cycle,
                       "first_control_vs_candidate": first_difference(baseline[key], candidate[key])}
                for label, data in (("control", baseline), ("candidate", candidate)):
                    greedy, proposed = data[cycle, case, 0], data[key]
                    row[label + "_first_vs_greedy"] = first_difference(greedy, proposed)
                    row[label + "_equal_through_stop"] = stopped(greedy, stops) == stopped(proposed, stops)
                    row[label + "_repeatable"] = proposed == data[0, case, depth]
                rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--stop-id", type=int, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.candidate, set(args.stop_id)), indent=2))
