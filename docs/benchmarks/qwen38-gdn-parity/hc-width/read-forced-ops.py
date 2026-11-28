from pathlib import Path
import json,sys
import numpy as np

def load(path):
    records=[]
    with path.open('rb') as f:
        while line:=f.readline():
            ly,phase,name,n=line.split()
            a=np.frombuffer(f.read(int(n)*4),dtype='<u4')
            if len(a)!=int(n):raise ValueError(path)
            records.append(((int(ly),phase.decode(),name.decode()),a))
    if len(records)!=48*(7+9):raise ValueError((path,len(records)))
    return records

d=Path(sys.argv[1])
data={}
for offset in (0,69):
    ref=load(d/f'o{offset}-b0.bin')
    for width in (1,2,3):
        candidate=load(d/f'o{offset}-b{width}.bin');out=[]
        for (key,a),(check,b) in zip(ref,candidate):
            if check!=key:raise ValueError(check)
            # The scalar QSA path uses a separate attention-output allocation.
            if key[0]%4==3 and key[2]=='gdn_out': continue
            n=np.count_nonzero(a!=b)
            if n:
                delta=None if key[-1]=='sel_ids10' else float(abs(a.view('<f4')-b.view('<f4')).max())
                out.append(dict(layer=key[0],phase=key[1],field=key[2],mismatches=int(n),maximum=delta))
        data[f'{offset}-{width}']={'nonidentical_buffers':len(out),'first':out[:8],
                                  'expert_id_differences':[o for o in out if o['field']=='sel_ids10'][:4]}
print(json.dumps(data,indent=2))
