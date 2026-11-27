"""Measure the unmodified, pinned SGLang runtime on one fixed Inkling request."""
import json
import os
import time
from pathlib import Path

def main():
    import sglang as sgl
    root=Path.cwd()
    cfg=dict(model_path=os.environ['OFFICIAL'],tp_size=1,ep_size=1,
             attention_backend='fa4',dtype='bfloat16',kv_cache_dtype='bf16',page_size=128,
             grammar_backend='xgrammar',context_length=8192,max_total_tokens=8192,
             max_running_requests=1,mem_fraction_static=0.85,cuda_graph_max_bs_decode=1,
             disable_radix_cache=True,enable_metrics=True)
    prompt=list(map(int,(root/'prompt.ids').read_text().strip().split(',')))
    assert len(prompt)==4096
    rows=[]
    engine=sgl.Engine(**cfg)
    try:
        for i in range(8):
            start=time.monotonic()
            resp=engine.generate(input_ids=prompt,sampling_params=dict(temperature=0,top_p=1,top_k=-1,min_p=0,sampling_seed=42,max_new_tokens=33,ignore_eos=True))
            elapsed=time.monotonic()-start
            ids=resp['output_ids'];meta=resp['meta_info']
            assert len(ids)==33 and all(0 <= token < 200058 for token in ids)
            assert meta['prompt_tokens']==4096 and meta['completion_tokens']==33 and meta['cached_tokens']==0
            assert meta['prefill_latency']>0 and meta['decode_throughput']>0
            rows.append(dict(rep=i,ids=ids,wall_ms=elapsed*1000,
                             prefill_ms=meta['prefill_latency']*1000,decode32_ms=32000/meta['decode_throughput']))
    finally:
        engine.shutdown()
    Path(os.environ['RESULT']).write_text(json.dumps(rows)+'\n')
if __name__=='__main__':main()
