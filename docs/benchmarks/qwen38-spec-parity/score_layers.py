import json
from pathlib import Path
import sys
import numpy as np
d=Path(sys.argv[1])
t=np.dtype([("pos","<u4"),("layer","<u4"),("stage","<u4"),("hc","<f4",10240),("routes","<i4",10)])
a,b=[np.memmap(d/f"{arm}.bin",dtype=t,mode="r") for arm in ("serial","verify")]
assert len(a)==len(b)==80*48*2
for k in ("pos","layer","stage"): assert np.array_equal(a[k],b[k])
rows=[]
for i,(x,y) in enumerate(zip(a,b)):
 v,w=x["hc"].astype(float),y["hc"].astype(float)
 baseline=np.linalg.norm(v); ref=list(map(int,x["routes"])); cand=list(map(int,y["routes"]))
 rows.append(dict(pos=int(x["pos"]),layer=int(x["layer"]),stage=int(x["stage"]),rel_l2=float(np.linalg.norm(v-w)/baseline),max_abs=float(np.abs(v-w).max()),same_order=ref==cand,same_set=set(ref)==set(cand),serial_experts=ref,verify_experts=cand))
levels=(0,1e-6,1e-5,1e-4,1e-3,1e-2,.05,.1)
cut=lambda r: {k:v for k,v in r.items() if not k.endswith("experts")}
summary={"first_above_relative_l2":[(thr,next((cut(r) for r in rows if r["rel_l2"]>thr),None)) for thr in levels],"first_changed_order":next((r for r in rows if not r["same_order"]),None),"first_changed_set":next((r for r in rows if not r["same_set"]),None),"final_token":[cut(r) for r in rows if r["pos"]==98],"changed_sets_by_token":{p:sum(r["pos"]==p and not r["same_set"] for r in rows) for p in range(19,99)}}
print(json.dumps(summary,indent=2))
