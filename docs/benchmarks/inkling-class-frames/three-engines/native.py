"""Native prefill/decode counters from the pinned vLLM build, one uncached request at a time."""
import json
import os
import time
from pathlib import Path

def main():
    from vllm import LLM, SamplingParams
    ids=[int(v) for v in Path('prompt.ids').read_text().strip().split(',')]
    assert len(ids)==1024
    llm=LLM(model=os.environ['OFFICIAL'],tensor_parallel_size=1,dtype='bfloat16',kv_cache_dtype='float16',
            max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=2048,
            kv_cache_memory_bytes=2304 * 1024**2,enable_prefix_caching=False,language_model_only=True,
            disable_log_stats=False,seed=42)
    generation=SamplingParams(temperature=0,max_tokens=33,ignore_eos=True,detokenize=False,seed=42)
    rows=[]
    for rep in range(8):
        start=time.perf_counter()
        (out,)=llm.generate([{'prompt_token_ids':ids}],generation,use_tqdm=False)
        wall=time.perf_counter()-start
        m=out.metrics; tokens=list(out.outputs[0].token_ids)
        assert len(tokens)==33 and out.prompt_token_ids==ids and out.num_cached_tokens==0
        assert m and m.num_generation_tokens==33 and m.num_preemptions==0
        assert 0<m.scheduled_ts<m.first_token_ts<m.last_token_ts and not m.is_corrupted
        rows.append(dict(rep=rep,ids=tokens,scheduled_ts=m.scheduled_ts,first_token_ts=m.first_token_ts,
                         last_token_ts=m.last_token_ts,prefill_ms=(m.first_token_ts-m.scheduled_ts)*1000,
                         decode_ms=(m.last_token_ts-m.first_token_ts)*1000,wall_ms=wall*1000))
    Path(os.environ['VLLM_LEG_RESULT']).write_text(json.dumps(rows)+'\n')

if __name__=='__main__': main()
