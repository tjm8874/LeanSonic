"""Capture full pre-sampling logits in a local, single-request quality run."""

import os
from pathlib import Path

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

VOCAB = 201024


class CaptureAndForce(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list):
        assert logits.shape == (1, VOCAB), tuple(logits.shape)
        (params,) = custom_param_list
        case = params["fixture_case"]
        assert case.isascii() and all(c.isalnum() or c == "_" for c in case)
        targets = params["target_ids"]
        path = Path(os.environ["INKLING_CAPTURE"]) / (case + ".f32")
        size = path.stat().st_size if path.exists() else 0
        position, remainder = divmod(size, VOCAB * 4)
        assert not remainder
        if position < len(targets):
            row = logits.detach().float().cpu().numpy().astype("<f4", copy=False)
            with path.open("ab") as stream:
                stream.write(row.tobytes())
            token = targets[position]
        else:
            token = 200006
        logits.fill_(float("-inf"))
        logits[0, token] = 0
        return logits
