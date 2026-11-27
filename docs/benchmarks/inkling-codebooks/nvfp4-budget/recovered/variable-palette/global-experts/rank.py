"""Rank expert restorations using training-only gain and measured whole-plane byte costs."""
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

official, cutoffs, routes, plan_file, budget_file, sampler = map(Path,sys.argv[1:])
plan=json.loads(plan_file.read_text())
control=json.loads(budget_file.read_text())
cuts=json.loads(cutoffs.read_text())["cutoffs"]
routing=json.loads(routes.read_text())["factor"]
offers={x["plane"]:x for x in control["offers"]}
assert hashlib.sha256(routes.read_bytes()).hexdigest()==plan["routing_sha256"]
assert hashlib.sha256(plan_file.read_bytes()).hexdigest()==control["plan_sha256"]
candidates=[]
with official.open('rb') as f,tempfile.TemporaryDirectory() as tmp:
    size,=struct.unpack('<Q',f.read(8));meta=json.loads(f.read(size));base=f.tell()
    for seed,key in enumerate(sorted(cuts),933):
        if key not in offers: continue
        gain=offers[key]["training_gain"]
        price=offers[key]["extra_bytes"]
        mode=plan["modes"][key]
        layer=int(key.split('.')[3]);width=meta[key+'_scale']['shape'][-1]
        wo=meta[key]['data_offsets'][0];so=meta[key+'_scale']['data_offsets'][0]
        a,b=meta[key+'_amax']['data_offsets'];f.seek(base+a);amplitude=struct.unpack('<128f',f.read(b-a))
        limit=Path(tmp)/'cut';limit.write_bytes(bytes(255 if x<0 else x for x in cuts[key]))
        env={**os.environ,'INKLING_OFFLINE_PALETTE':plan["palettes"][key]}
        rows=subprocess.check_output([sampler,official,str(base+wo),str(base+so),str(width),limit,str(seed)],text=True,env=env).splitlines()
        assert len(rows)==131
        energy=0.;benefits=[]
        for e,row in enumerate(rows[:128]):
            name,index,signal,err0,errlo,errall=row.split()
            assert name=='expert' and int(index)==e
            amp=amplitude[e]**2
            energy+=int(signal)*amp
            benefit=(int(errall if mode=='all' else errlo)-int(err0))*amp*routing[layer][e]
            benefits.append(benefit)
        m=next(row for row in rows[128:] if row.split()[0]==mode)
        norm=int(m.split()[3])*(3 if 6<=layer<18 else 1)/energy
        scaled=[v*norm for v in benefits]
        assert math.isclose(sum(scaled),gain,rel_tol=2e-10),(key,sum(scaled),gain)
        assert price>0
        for e,benefit in enumerate(scaled):
            if benefit>0:candidates.append(dict(plane=key,expert=e,gain=benefit,estimated_bytes=price/128))
candidates.sort(key=lambda x:(-x["gain"]/x["estimated_bytes"],x["plane"],x["expert"]))
print(json.dumps(dict(rank=candidates,whole_plane_gain=control['train_objective_parent']-control['train_objective_candidate'],
                      approach='per-plane average byte cost, energy-normalized training gain per expert',
                      parent_objective=control['train_objective_parent'],parent_bytes=control['parent_bytes'],cap=control['cap'],
                      plan_sha256=control['plan_sha256'],training_sha256=plan['routing_sha256']),indent=2))
