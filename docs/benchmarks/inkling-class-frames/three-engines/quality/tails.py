"""Rank matched differences against the official vLLM runtime using complete BF16 scores."""
import csv,json,statistics,sys
from pathlib import Path
split=sys.argv[1]
def load(arm):
 d=Path(split+'-'+arm);assert (d/'done').exists()
 with (d/'scored.rows').open() as f:return list(csv.reader(f))
lean,vllm=load('lean'),load('vllm');assert len(lean)==len(vllm)==25500
pairs=[]
for a,b in zip(lean,vllm):
 assert a[:3]==b[:3]
 pairs.append(dict(chunk=int(a[0]),position=int(a[1]),target=int(a[2]),lean_top=int(a[3]),vllm_top=int(b[3]),lean_kl=float(a[4]),vllm_kl=float(b[4]),lean_agrees=a[6]=='true',vllm_agrees=b[6]=='true',difference=float(a[4])-float(b[4])))
ranked=sorted(pairs,key=lambda p:p['difference'],reverse=True)
delta=[row['difference'] for row in ranked]
print(json.dumps(dict(positions=len(pairs),mean_kl_difference=statistics.mean(delta),lean_kl_more_by_1=sum(d>1 for d in delta),vllm_kl_more_by_1=sum(d < -1 for d in delta),top1=dict(both=sum(x['lean_agrees'] and x['vllm_agrees'] for x in pairs),lean_only=sum(x['lean_agrees'] and not x['vllm_agrees'] for x in pairs),vllm_only=sum(x['vllm_agrees'] and not x['lean_agrees'] for x in pairs),neither=sum(not x['lean_agrees'] and not x['vllm_agrees'] for x in pairs)),largest_lean_excess=ranked[:10],largest_vllm_excess=ranked[-10:][::-1]),indent=2))
