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
    parser.add_argument("--serial", action="store_true",
                        help="prefill 256 tokens then teacher-force positions 256..510 one by one")
    parser.add_argument("--save-kld", type=Path, help="also save the observed schedule as a KLD reference")
    args = parser.parse_args()
    if args.serial and (args.batch != 512 or args.ubatch != 512):
        parser.error("--serial requires --batch 512 --ubatch 512")
    if args.save_kld and args.save_kld.resolve() == args.ref.resolve():
        parser.error("--save-kld must differ from the input reference")
    with args.ref.open("rb") as ref:
        magic, ctx, vocab, chunks = struct.unpack("<8sIII", ref.read(20))
        if (magic, ctx, vocab) != (b"_logits_", 512, 200058) or not 0 < args.chunks <= chunks:
            parser.error("expected an Inkling 512-token reference covering the requested chunks")
        expected = ref.read(args.chunks * ctx * 4)

    # Stage on the destination filesystem so a failed run leaves no partial reference.
    with tempfile.TemporaryDirectory(dir=args.save_kld.parent if args.save_kld else None) as folder:
        ids = Path(folder) / "ids.bin"
        layout = Path(folder) / "layout.txt"
        staged = Path(folder) / "observed.kld"
        mode = os.environ.copy()
        mode.pop("LEANSONIC_BF16_SERIAL_SCORE", None)
        mode.pop("LEANSONIC_BF16_SERIAL_LAYOUT", None)
        if args.serial:
            mode.update(LEANSONIC_BF16_SERIAL_SCORE="1", LEANSONIC_BF16_SERIAL_LAYOUT=str(layout))
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
                         "-ngl", "99", "-ctk", "f16", "-ctv", "f16", "-t", "8", "--no-warmup",
                         "--chunks", str(args.chunks), *(["--save-all-logits", staged] if args.save_kld else [])],
                        env={**mode, "LEANSONIC_BF16_ROWS": f"/dev/fd/{write_fd}",
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
        if args.serial:
            expected_layout = [(0, ctx // 2)] + [(pos, 1) for pos in range(ctx // 2, ctx - 1)]
            if not layout.is_file():
                raise SystemExit("llama.cpp did not record the serial decode schedule")
            observed = [tuple(map(int, line.split())) for line in layout.read_text().splitlines()]
            if observed != expected_layout:
                raise SystemExit("unexpected llama.cpp serial decode schedule")
        if args.save_kld:
            header = struct.pack("<8sIII", magic, ctx, vocab, args.chunks)
            rows = result["rows"]
            with staged.open("rb") as saved:
                if saved.read(len(header)) != header or saved.read(len(expected)) != expected:
                    raise SystemExit("saved KLD header or chunk tokens differ from the reference")
            if staged.stat().st_size != len(header) + len(expected) + rows * (8 + 2 * vocab):
                raise SystemExit("unexpected size of saved KLD")
            staged.replace(args.save_kld)
        print(output, end="")


if __name__ == "__main__":
    main()
