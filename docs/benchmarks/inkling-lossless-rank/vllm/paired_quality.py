"""Pair complete full-corpus score rows from LeanSonic rank and vLLM."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

def parse(path):
    raw=path.read_bytes()
    rows=list(csv.reader(raw.decode().splitlines()))
    assert len(rows)==25500
    return rows,hashlib.sha256(raw).hexdigest()

def analyze(rank, vllm):
    a, ah = parse(rank)
    b, bh = parse(vllm)
    chunks=[[0.,0.,0.] for _ in range(100)]
    rows=[]
    agreement={'both':0,'rank_only':0,'vllm_only':0,'neither':0}
    for i,(x,y) in enumerate(zip(a,b,strict=True)):
        assert x[:3]==y[:3] and x[:2]==[str(i//255),str(256+i%255)]
        kl=float(x[4])-float(y[4])
        nll=float(x[5])-float(y[5])
        mx,my=x[6]=='true',y[6]=='true'
        agreement['both' if mx and my else 'rank_only' if mx else 'vllm_only' if my else 'neither']+=1
        sums=chunks[i//255]
        sums[0]+=kl
        sums[1]+=nll
        sums[2]+=int(mx)-int(my)
        rows.append(dict(chunk=i//255,position=256+i%255,target=int(x[2]),rank_top=int(x[3]),vllm_top=int(y[3]),rank_kl=float(x[4]),vllm_kl=float(y[4]),delta=kl,rank_matches=mx,vllm_matches=my))
    ordered=sorted(rows,key=lambda s:s['delta'])
    rng=random.Random(382)
    draws=[[],[],[]]
    def estimate(sample):
        mkl,logppl,top=map(sum,zip(*sample))
        return (mkl/25500,math.expm1(logppl/25500),top/25500)
    for _ in range(10000):
        for acc,stat in zip(draws,estimate(rng.choices(chunks,k=100))):
            acc.append(stat)
    effects={}
    for name,absolute,draw in zip(('mean_kl','relative_ppl','top1_fraction'),estimate(chunks),draws):
        draw.sort()
        effects[name]={'rank_minus_vllm':absolute,'percentile_95':[draw[249],draw[9749]]}
    return dict(rows=25500,rank_sha256=ah,vllm_sha256=bh,effects=effects,agreement=agreement,
        chunks_lower_mean_kl=sum(row[0]<0 for row in chunks),
        kl_exceeds_vllm_by_one=sum(row['delta']>1 for row in rows),
        kl_below_vllm_by_one=sum(row['delta'] < -1 for row in rows),
        worst_relative=ordered[-10:][::-1],best_relative=ordered[:10])
if __name__=='__main__':
    print(json.dumps(analyze(*map(Path,sys.argv[1:])),indent=2))
