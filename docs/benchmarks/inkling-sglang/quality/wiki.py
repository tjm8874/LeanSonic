"""Replay the exact oracle token histories and feed the native corpus scorer one chunk at a time."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import sglang as sgl
from trace_processor import CaptureAndForce


def main():
    reference, binary, split = sys.argv[1:]
    with open(reference, "rb") as f:
        magic, ctx, vocab, count = struct.unpack("<8sIII", f.read(20))
        assert (magic, ctx, vocab) == (b"_logits_", 512, 200058) and count >= 100
        tokens = np.frombuffer(f.read(100 * ctx * 4), dtype="<u4").reshape(100, ctx)
    work = Path(os.environ["INKLING_CAPTURE"])
    work.mkdir()
    config = json.loads(Path("functional.json").read_text())["config"]
    engine = sgl.Engine(
        model_path=os.environ["OFFICIAL"],
        enable_custom_logit_processor=True,
        disable_overlap_schedule=True,
        **config,
    )
    try:
        with open(split + ".tmp", "xb") as summary:
            scorer = subprocess.Popen(
                [
                    binary,
                    "eval-kld",
                    "--ref",
                    reference,
                    "--logits-f32",
                    "-",
                    "--chunks",
                    "100",
                ],
                stdin=subprocess.PIPE,
                stdout=summary,
            )
            try:
                for n, chunk in enumerate(tokens):
                    case = f"{split}_{n:03d}"
                    ids = chunk.tolist()
                    assert max(ids) < vocab
                    answer = ids[257:]
                    response = engine.generate(
                        input_ids=ids[:257],
                        sampling_params={
                            "temperature": 0,
                            "ignore_eos": True,
                            "max_new_tokens": 255,
                            "custom_params": {
                                "fixture_case": case,
                                "target_ids": answer,
                            },
                        },
                        custom_logit_processor=CaptureAndForce.to_str(),
                    )
                    assert (
                        response["output_ids"] == answer
                        and response["meta_info"]["prompt_tokens"] == 257
                    )
                    path = work / (case + ".f32")
                    rows = np.memmap(path, dtype="<f4", mode="r").reshape(255, 201024)
                    assert np.isneginf(rows[:, vocab:]).all()
                    for row in rows:
                        scorer.stdin.write(row[:vocab].tobytes())
                    del rows
                    path.unlink()
                    print(case, flush=True)
                scorer.stdin.close()
                if scorer.wait():
                    raise RuntimeError("native eval-kld failed")
            finally:
                if scorer.poll() is None:
                    scorer.kill()
                    scorer.wait()
    finally:
        engine.shutdown()
    v = json.loads(Path(split + ".tmp").read_text())
    assert v["rows"] == 25500 and v["chunks"] == 100
    Path(split + ".tmp").rename(split + ".json")


if __name__ == "__main__":
    main()
