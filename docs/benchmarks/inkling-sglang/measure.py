"""Generate the independent bilingual fixture on the installed public SGLang engine."""

import json
import os
import time
from pathlib import Path

import sglang as sgl


def run():
    model = Path(os.environ["OFFICIAL"])
    rng = {
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "min_p": 0,
        "sampling_seed": 42,
        "skip_special_tokens": False,
    }
    rows = json.loads(Path("fixtures.json").read_text())
    dist = json.loads(Path(os.environ["SGLANG_PREPARED"]).read_text())
    weights = {c["path"]: (model / c["path"]).stat().st_size for c in dist["files"]}
    assert all(weights[c["path"]] == c["bytes"] for c in dist["files"])
    config = {
        "model_path": str(model),
        "tp_size": 1,
        "ep_size": 1,
        "attention_backend": "fa4",
        "dtype": "bfloat16",
        "page_size": 128,
        "grammar_backend": "xgrammar",
        "context_length": 2048,
        "max_total_tokens": 2048,
        "max_running_requests": 1,
        "mem_fraction_static": 0.85,
        "cuda_graph_max_bs_decode": 1,
        "disable_radix_cache": True,
        "enable_metrics": True,
    }
    engine = sgl.Engine(**config)
    try:
        meta = {
            "model_revision": dist["model_revision"],
            "engine": dist["sglang_commit"],
            "distribution_files": weights,
            "config": {k: v for k, v in config.items() if k != "model_path"},
            "records": [],
        }
        records = meta["records"]
        outdir = Path("traces")
        outdir.mkdir()
        for row in rows + [rows[0]]:
            name = row["id"]
            ids = list(map(int, (Path("ids") / (name + ".ids")).read_text().split()))
            start = time.monotonic()
            response = engine.generate(
                input_ids=ids,
                sampling_params=dict(**rng, max_new_tokens=384, ignore_eos=False),
            )
            duration = time.monotonic() - start
            info = response["meta_info"]
            assert (
                info["prompt_tokens"] == len(ids) and info.get("cached_tokens", 0) == 0
            )
            tokens = list(response["output_ids"])
            finish = info["finish_reason"]["type"]
            assert finish in ("stop", "length") and len(tokens) > 0
            if finish == "stop" and tokens[-1] != 200006:
                assert info["finish_reason"].get("matched") == 200006, info[
                    "finish_reason"
                ]
                tokens.append(200006)
            assert len(tokens) <= 384 and (tokens[-1] == 200006 or len(tokens) == 384)
            with (outdir / "sglang.ids").open("a") as f:
                f.write(name + " " + " ".join(map(str, tokens)) + "\n")
            records.append(
                {
                    "case": name,
                    "usage": info["completion_tokens"],
                    "finish": finish,
                    "wall_seconds": duration,
                }
            )
        assert (outdir / "sglang.ids").read_text().splitlines()[0] == (
            outdir / "sglang.ids"
        ).read_text().splitlines()[-1]
        Path("functional.json").write_text(json.dumps(meta, indent=2) + "\n")
        ids = list(
            map(int, Path("inkling-readme1024.ids").read_text().strip().split(","))
        )
        assert len(ids) == 1024
        timing = []
        for _ in range(8):
            start = time.monotonic()
            response = engine.generate(
                input_ids=ids,
                sampling_params=dict(**rng, max_new_tokens=33, ignore_eos=True),
            )
            wall = time.monotonic() - start
            info = response["meta_info"]
            tokens = response["output_ids"]
            assert (
                len(tokens) == 33
                and info["prompt_tokens"] == 1024
                and info["completion_tokens"] == 33
                and info.get("cached_tokens", 0) == 0
            )
            if timing:
                assert tokens == timing[0]["predictions"]
            timing.append(
                {
                    "predictions": tokens,
                    "elapsed": wall,
                    "metrics": {
                        k: v
                        for k, v in info.items()
                        if any(
                            t in k
                            for t in ("latency", "throughput", "tokens", "duration")
                        )
                    },
                }
            )
        Path("timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    run()
