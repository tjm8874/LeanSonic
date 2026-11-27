"""Score fixed token histories in the offline SGLang Engine, one case at a time."""

import json
import os
from pathlib import Path

import sglang as sgl
from trace_processor import CaptureAndForce


def main():
    fixture = Path(os.environ["INKLING_FIXTURE"])
    out = Path(os.environ["INKLING_CAPTURE"])
    out.mkdir()
    config = json.loads(Path("functional.json").read_text())["config"]
    engine = sgl.Engine(
        model_path=os.environ["OFFICIAL"],
        enable_custom_logit_processor=True,
        disable_overlap_schedule=True,
        **config,
    )
    try:
        for case in (fixture / "order.txt").read_text().split():
            prefix = list(map(int, (fixture / case / "prefix.ids").read_text().split()))
            answer = list(map(int, (fixture / case / "answer.ids").read_text().split()))
            response = engine.generate(
                input_ids=prefix,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": len(answer),
                    "ignore_eos": True,
                    "custom_params": {"fixture_case": case, "target_ids": answer},
                },
                custom_logit_processor=CaptureAndForce.to_str(),
            )
            assert list(response["output_ids"]) == answer
            assert response["meta_info"]["prompt_tokens"] == len(prefix)
            assert (out / (case + ".f32")).stat().st_size == 4 * 201024 * len(answer)
            print(case, len(answer), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
