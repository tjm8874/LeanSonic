#!/usr/bin/env python3
"""Compare the full greedy continuation across speculative widths using the normal CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def generate(engine, weights, tokens, width, length):
    command = [str(engine), "bench", "--weights", str(weights), "--tokens", "@" + str(tokens),
               "--spec", str(width), "--gen", str(length), "--repeat", "1"]
    proc = subprocess.run(command, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    proc.check_returncode()
    mode = f"DSpark(K={width})" if width else "off"
    if f"| Spec: {mode} |" not in proc.stderr:
        raise ValueError(f"expected the GPU loader to report {mode}")
    try:
        ids = [int(value) for value in proc.stdout.splitlines()[-1].split(",")]
    except (IndexError, ValueError) as error:
        raise ValueError(f"spec={width}: missing token list in benchmark output") from error
    if len(ids) != length:
        raise ValueError(f"spec={width}: requested {length} tokens but received {len(ids)}")
    return ids


def digest(ids):
    line = ",".join(map(str, ids)) + "\n"
    return hashlib.sha256(line.encode("ascii")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--spec", type=int, required=True)
    parser.add_argument("--gen", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.spec <= 7 or args.gen < 1:
        parser.error("--spec must be between 1 and 7, and --gen must be positive")
    engine = args.engine.resolve()
    prompt_sha = hashlib.sha256(args.tokens.read_bytes()).hexdigest()
    regular = generate(engine, args.weights, args.tokens, 0, args.gen)
    speculative = generate(engine, args.weights, args.tokens, args.spec, args.gen)
    first = next((i for i, pair in enumerate(zip(regular, speculative)) if pair[0] != pair[1]), None)
    print(json.dumps({
        "prompt_sha256": prompt_sha, "generated": args.gen, "spec_width": args.spec,
        "regular_sha256": digest(regular), "speculative_sha256": digest(speculative),
        "first_difference_zero_based": first,
        "regular_token": None if first is None else regular[first],
        "speculative_token": None if first is None else speculative[first],
    }))
    return int(first is not None)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
