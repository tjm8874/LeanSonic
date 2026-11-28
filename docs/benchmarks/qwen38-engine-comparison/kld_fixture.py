"""Write llama.cpp's uint16 log-probability reference for Basalt's native PPL gate."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def write_base(fixtures, reference, dest):
    seqs = [list(map(int, line.split())) for line in fixtures.read_text().splitlines()]
    context = min(map(len, seqs))
    if context < 4:
        raise ValueError("need at least four input tokens in each case")
    vocab = 248320
    rows = np.loadtxt(reference / "rows.tsv", dtype=np.int64, ndmin=2)
    logits = np.memmap(reference / "logits.f32", dtype="<f4", mode="r").reshape(
        len(rows), vocab
    )
    lookup = {(int(case), int(pos)): i for i, (case, pos, *_) in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("duplicate positions")
    worst = 0.0
    max_tail = 0.0
    expected = []
    with dest.open("xb") as out:
        out.write(b"_logits_" + struct.pack("<Iii", context, vocab, len(seqs)))
        for s in seqs:
            out.write(np.asarray(s[:context], dtype="<i4").tobytes())
        for case, s in enumerate(seqs):
            for pos in range(context // 2, context - 1):
                i = lookup[case, pos]
                r = rows[i]
                if tuple(r[2:4]) != tuple(s[pos : pos + 2]):
                    raise ValueError("unaligned target or prefix")
                v = logits[i].astype(np.float64)
                v -= v.max()
                v -= np.log(np.exp(v).sum())
                floor = np.float32(v.min())
                step = np.float32((v.max() - v.min()) / 65535)
                if not np.isfinite(v).all() or step <= 0:
                    raise ValueError("invalid row")
                codes = np.clip(np.rint((v - floor) / step), 0, 65535).astype("<u2")
                if int(codes.argmax()) != int(r[4]):
                    raise ValueError("export changes top token")
                approx = floor + step * codes.astype(np.float32)
                worst = max(worst, float(np.abs(approx - v).max()))
                max_tail = max(max_tail, float(np.exp(v[v <= -16]).sum()))
                out.write(struct.pack("<ff", step, floor))
                out.write(codes.tobytes())
                expected.append([case, pos, int(r[3]), int(r[4])])
    return {
        "context": context,
        "vocabulary": vocab,
        "rows": expected,
        "max_abs_logprob_error": worst,
        "max_reference_mass_at_or_below_minus_16": max_tail,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fixtures", "reference", "dest"):
        parser.add_argument(key, type=Path)
    a = parser.parse_args()
    print(json.dumps(write_base(a.fixtures, a.reference, a.dest), indent=2))
