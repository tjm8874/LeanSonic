import csv,json
from pathlib import Path
rows=[]
with open('kernels.tsv') as f:
    for a,b,name,gx,gy,gz,bx,by,bz,reg,static,dynamic,node in csv.reader(f,delimiter='\t'):
        rows.append((int(a),int(b),name,int(node),int(reg),int(dynamic)))
assert 'dropped=0' in Path('capture.log').read_text()
out=[]
for line in Path('ranges').read_text().splitlines():
    run,a,b,c,ids=line.split();run,a,b,c=map(int,(run,a,b,c))
    assert a<b<c and len(ids.split(','))==32
    rec=dict(run=run,predictions=list(map(int,ids.split(','))),phases={})
    for kind,lo,hi in [('prefill',a,b),('decode32',b,c)]:
        selected=sorted(row for row in rows if lo<=row[0]<hi)
        assert len(selected)>100 and all(s<e<=hi for s,e,*_ in selected)
        totals={};end=0;union=0
        for s,e,name,node,reg,smem in selected:
            union+=max(0,e-max(s,end));end=max(end,e)
            g=totals.setdefault(name,[0,0,0,0]);g[0]+=1;g[1]+=e-s;g[2]=max(g[2],reg);g[3]=max(g[3],smem)
        if kind=='decode32':
            assert sum(n for name,(n,*_) in totals.items() if 'argmax' in name)>=32
        rec['phases'][kind]=dict(kernels=len(selected),graph_nodes=sum(bool(r[3]) for r in selected),span_ms=(end-selected[0][0])/1e6,union_ms=union/1e6,wall_ms=(hi-lo)/1e6,by_kernel={name:dict(calls=g[0],total_ms=g[1]/1e6,registers=g[2],dynamic_shared=g[3]) for name,g in sorted(totals.items(),key=lambda x:-x[1][1])})
    out.append(rec)
assert [r['run'] for r in out]==[0,1,2] and out[0]['predictions']==out[1]['predictions']==out[2]['predictions']
print(json.dumps(out,indent=2))
