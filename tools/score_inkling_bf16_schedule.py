#!/usr/bin/env python3
"""Score llama-perplexity's FP32 rows against an Inkling KLD reference on Linux."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("perplexity", "engine", "model", "text", "ref"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--ubatch", type=int, required=True)
    parser.add_argument("--chunks", type=int, default=100)
    args = parser.parse_args()
    with args.ref.open("rb") as ref:
        magic, ctx, vocab, chunks = struct.unpack("<8sIII", ref.read(20))
        if (magic, ctx, vocab) != (b"_logits_", 512, 200058) or not 0 < args.chunks <= chunks:
            parser.error("expected an Inkling 512-token reference covering the requested chunks")
        expected = ref.read(args.chunks * ctx * 4)

    with tempfile.TemporaryDirectory() as folder:
        ids = Path(folder) / "ids.bin"
        # An anonymous pipe avoids materializing 20 GB of logits per 100-chunk run.
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb") as reader, os.fdopen(write_fd, "wb") as writer:
            with subprocess.Popen(
                [args.engine.resolve(), "eval-kld", "--logits-f32", "-", "--ref", args.ref,
                 "--chunks", str(args.chunks)], stdin=reader, stdout=subprocess.PIPE, text=True,
            ) as scorer:
                reader.close()
                try:
                    subprocess.run(
                        [args.perplexity.resolve(), "-m", args.model, "-f", args.text,
                         "-c", str(ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
                         "-ngl", "99", "-t", "8", "--no-warmup", "--chunks", str(args.chunks)],
                        env={**os.environ, "LEANSONIC_BF16_ROWS": f"/dev/fd/{write_fd}",
                             "LEANSONIC_BF16_IDS": str(ids)},
                        pass_fds=(write_fd,), stdout=subprocess.DEVNULL, check=True,
                    )
                finally:
                    writer.close()
                output, _ = scorer.communicate()
                if scorer.returncode:
                    raise SystemExit(f"LeanSonic scorer exited with status {scorer.returncode}")

        if ids.read_bytes() != expected:
            raise SystemExit("captured chunk token IDs differ from the reference")
        result = json.loads(output)
        if result["rows"] != args.chunks * (ctx - 1 - ctx // 2):
            raise SystemExit("unexpected number of scored rows")
        print(output, end="")


if __name__ == "__main__":
    main()
