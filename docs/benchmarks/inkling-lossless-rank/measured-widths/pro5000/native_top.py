"""Check native greedy top choices on the common prefix on each GPU."""
import json
import os
from pathlib import Path
def main():
    from vllm import LLM,SamplingParams
    ids=list(map(int,Path('prompt.ids').read_text().strip().split(',')))
    expected=list(map(int,Path('expected.ids').read_text().strip().split(',')))
    assert len(ids)==1024 and len(expected)==32
    model=LLM(model=os.environ['OFFICIAL'],tensor_parallel_size=1,dtype='bfloat16',kv_cache_dtype='float16',
              max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=2048,kv_cache_memory_bytes=2304*1024**2,
              enable_prefix_caching=False,language_model_only=True,disable_log_stats=False,seed=42)
    results=[]
    for rep,logs in enumerate([None,8,None,8]):
        params=SamplingParams(temperature=0,max_tokens=33,ignore_eos=True,detokenize=False,seed=42,logprobs=logs)
        (result,)=model.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)
        out=result.outputs[0];tokens=list(out.token_ids)
        assert len(tokens)==33 and result.num_cached_tokens==0
        row=dict(rep=rep,requested_logprobs=logs,ids=tokens, matches_timed_anchor=tokens[:32]==expected)
        if logs is not None:
            row['fourth_top']={str(k):dict(logprob=float(v.logprob),rank=v.rank) for k,v in out.logprobs[3].items()}
            row['agrees_with_unobserved']=tokens==results[-1]['ids']
        results.append(row)
    Path('result.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
