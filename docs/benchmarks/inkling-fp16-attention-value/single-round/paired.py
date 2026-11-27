"""Paired uncertainty by whole WikiText chunks for two recurrence changes."""
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

def analyze(current_file,candidate_file):
    def parse(file):
        raw=Path(file).read_bytes()
        rows=list(csv.reader(raw.decode().splitlines()))
        assert len(rows)==25500
        return rows, hashlib.sha256(raw).hexdigest()
    a,ha=parse(current_file);b,hb=parse(candidate_file)
    by_chunk=[[0.,0.,0.] for _ in range(100)]
    top_before=top_after=0
    tails=[]
    for index,(x,y) in enumerate(zip(a,b,strict=True)):
        chunk=index//255
        assert x[:3]==y[:3] and x[:2]==[str(chunk),str(256+index%255)]
        kl=float(y[4])-float(x[4]);nll=float(y[5])-float(x[5])
        first=x[6]=='true';second=y[6]=='true'
        by_chunk[chunk][0]+=kl;by_chunk[chunk][1]+=nll;by_chunk[chunk][2]+=second-first
        tails.append((kl,chunk,int(x[1])));top_before+=first;top_after+=second
    totals=[sum(col) for col in zip(*by_chunk)]
    rand=random.Random(85523)
    values=[[],[],[]]
    for _ in range(10000):
        draw=rand.choices(by_chunk,k=100)
        for i,col in enumerate(zip(*draw)):values[i].append(sum(col)/25500)
    def score(i):
        samples=sorted(values[i])
        transform=math.expm1 if i==1 else (lambda x:x)
        return dict(delta=transform(totals[i]/25500),interval95=[transform(samples[249]),transform(samples[9749])])
    largest,chunk,pos=min(tails)
    return dict(current_rows_sha256=ha,candidate_rows_sha256=hb,mean_kl=score(0),ppl_relative=score(1),top1_fraction=score(2),
                recovered_net=top_after-top_before,largest_improvement=dict(chunk=chunk,position=pos,delta_kl=largest),
                change_excluding_largest_row=(totals[0]-largest)/25499,
                change_excluding_its_entire_chunk=(totals[0]-by_chunk[chunk][0])/(99*255),
                lower_mean_kl_chunks=sum(v[0]<0 for v in by_chunk),higher_mean_kl_chunks=sum(v[0]>0 for v in by_chunk),
                per_chunk_sum_deltas=by_chunk)
if __name__=='__main__':
    print(json.dumps(analyze(*sys.argv[1:]),indent=2))
