"""Capture this request's pre-sampling logits before forcing each reference token."""
import os
from pathlib import Path

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


class CaptureAndForce(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.slots = {}

    def add_request(self, req_idx, sampling_params):
        config = (sampling_params.extra_args or {}).get("oracle_capture")
        self.slots.pop(req_idx, None)
        if config is None:
            return False
        name, prompt_len, targets = config
        assert name in ("test", "valid") and prompt_len == 257 and len(targets) == 255
        self.slots[req_idx] = (name, targets, 0)
        return True

    def apply(self, logits, ctx):
        import numpy as np
        import torch
        if not self.slots:
            return logits
        indices = ctx.expanded_idx_mapping.tolist()
        positions = ctx.pos.tolist()
        assert len(indices) == len(positions) == 1
        slot, = indices
        if slot not in self.slots:
            return logits
        name, targets, step = self.slots[slot]
        assert positions == [256 + step] and logits.dtype == torch.float32
        row = logits[0].detach().cpu().numpy()
        # An engine may trim the vocabulary itself. Reject finite extra columns.
        assert row.size >= 200058 and np.isneginf(row[200058:]).all()
        row = row[:200058]
        assert not np.isnan(row).any() and np.isfinite(row).any() and not np.isposinf(row).any()
        base = Path(os.environ["VLLM_CAPTURE_DIR"])
        with (base / (name + ".rows")).open("xb" if step == 0 else "ab") as stream:
            stream.write(row.astype("<f4", copy=False).tobytes())
        if step == 0:
            (base / (name + ".top")).write_text(str(int(np.argmax(row))))
        logits[0].fill_(float("-inf"))
        logits[0, targets[step]] = 0
        if step + 1 == len(targets):
            del self.slots[slot]
        else:
            self.slots[slot] = (name, targets, step + 1)
        return logits
