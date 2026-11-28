"""Split a BF16 teacher dump by input length for the native KLD scorers."""
import argparse
from pathlib import Path

import numpy as np
from kld_fixture import write_base


def split(fixtures, reference, dest):
    rows = np.loadtxt(reference / "rows.tsv", dtype=np.int64, ndmin=2)
    logits = np.memmap(reference / "logits.f32", dtype="<f4", mode="r", shape=(len(rows), 248320))
    for case, line in enumerate(fixtures.read_text().splitlines()):
        tokens = list(map(int, line.split()))
        selected = rows[:, 0] == case
        info = rows[selected].copy()
        if (info[:, 1].tolist() != list(range(len(tokens) - 1))
                or info[:, 2].tolist() != tokens[:-1]
                or info[:, 3].tolist() != tokens[1:]):
            raise ValueError(f"unaligned teacher data for case {case}")
        work = dest / str(case)
        work.mkdir()
        (work / "fixtures.ids").write_text(line + "\n")
        info[:, 0] = 0
        np.savetxt(work / "rows.tsv", info, fmt="%d", delimiter="\t")
        logits[selected].tofile(work / "logits.f32")
        write_base(work / "fixtures.ids", work, work / "ref.kld")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixtures", "reference", "new_directory"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.new_directory.mkdir()
    split(args.fixtures, args.reference, args.new_directory)
