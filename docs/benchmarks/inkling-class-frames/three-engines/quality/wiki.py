import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


def main():
    import numpy as np
    from vllm import LLM, SamplingParams
    ref, binary, split = sys.argv[1:]
    assert split in ("test", "valid")
    with open(ref, "rb") as f:
        magic, ctx, vocab, count = struct.unpack("<8sIII", f.read(20))
        assert (magic, ctx, vocab) == (b"_logits_", 512, 200058) and count >= 100
        ids = np.frombuffer(f.read(100 * ctx * 4), dtype="<u4").reshape(100, ctx)
    model = LLM(model=os.environ["OFFICIAL"], dtype="bfloat16", kv_cache_dtype="float16", language_model_only=True,
                tensor_parallel_size=1, max_model_len=2048, max_num_seqs=1,
                max_num_batched_tokens=2048, kv_cache_memory_bytes=2304 * 1024**2,
                enable_prefix_caching=False, seed=42, logits_processors=["capture:CaptureAndForce"])
    common = dict(temperature=0, detokenize=False, ignore_eos=True, seed=42)
    tmp = Path(split+".tmp")
    with tmp.open("xb") as summary:
        scorer = subprocess.Popen([binary, "eval-kld", "--ref", ref, "--logits-f32", "-", "--chunks", "100"],
                                  stdin=subprocess.PIPE, stdout=summary)
        try:
            for index, chunk in enumerate(ids):
                prefix = chunk[:257].tolist()
                targets = chunk[257:].tolist()
                prompt = [{"prompt_token_ids": prefix}]
                if index == 0:
                    free, = model.generate(prompt, SamplingParams(max_tokens=1, **common), use_tqdm=False)
                params = SamplingParams(max_tokens=255, extra_args={"oracle_capture":[split,257,targets]}, **common)
                out, = model.generate(prompt, params, use_tqdm=False)
                assert list(out.outputs[0].token_ids) == targets and list(out.prompt_token_ids) == prefix
                path = Path(os.environ["VLLM_CAPTURE_DIR"]) / (split+".rows")
                assert path.stat().st_size == 255*200058*4
                if index == 0:
                    top = int(path.with_suffix(".top").read_text())
                    assert list(free.outputs[0].token_ids) == [top]
                with path.open("rb") as rows:
                    shutil.copyfileobj(rows, scorer.stdin, 1<<20)
                path.unlink()
                print(split,index,flush=True)
            scorer.stdin.close()
            if scorer.wait():
                raise RuntimeError("CPU oracle scorer failed")
        finally:
            if scorer.poll() is None:
                scorer.kill()
                scorer.wait()
    result=json.loads(tmp.read_text())
    assert result["rows"]==25500 and result["chunks"]==100
    tmp.rename(split+".json")

if __name__ == "__main__":
    main()
