"""Price all remaining pure-ternary down projections and compute a whole-plane training control."""
import hashlib,itertools,json,struct,sys
from pathlib import Path
low,high,plan_file,sample_file,embed_receipt=map(Path,sys.argv[1:])
def header(p):
 with (p/'model-00001-of-00001.safetensors').open('rb') as f:
  n,=struct.unpack('<Q',f.read(8));return json.loads(f.read(n))
def size(meta,key):
 return sum(meta[k]['data_offsets'][1]-meta[k]['data_offsets'][0] for k in (key+'.nvscale_four_v1',key+'_scale.mean8_v1'))
parent=sum(f.stat().st_size for f in low.rglob('*') if f.is_file());assert parent==7763991646
embed=json.loads(embed_receipt.read_text());saving=embed['parent_bytes']-embed['package_bytes'];cap=7764291776+saving
lm,hm=header(low),header(high);assert set(lm)==set(hm)
plan=json.loads(plan_file.read_text());samples=json.loads(sample_file.read_text());offers=[]
risk=0
for k in sorted(plan['modes']):
 if not k.endswith('.w2_weight') or '.layers.17.' in k:continue
 assert plan['modes'][k]=='all' and plan['palettes'][k]=='06'
 a,b=samples[k]['all']['weighted_cost'],samples[k]['control']['weighted_cost'];risk+=a
 extra=size(hm,k)-size(lm,k)
 assert extra>0 and a>b
 offers.append(dict(plane=k,extra_bytes=extra,training_gain=a-b))
assert len(offers)==19
front=[(0,0,())]
for v in offers:
 added=[(sz+v['extra_bytes'],g+v['training_gain'],sel+(v['plane'],)) for sz,g,sel in front if sz+v['extra_bytes']<=cap-parent-4096]
 best=-1;kept=[]
 for sz,g,sel in sorted(front+added,key=lambda x:(x[0],-x[1])):
  if g>best:kept.append((sz,g,sel));best=g
 front=kept
chosen=front[-1]
print(json.dumps(dict(cap=cap,parent_bytes=parent,plan_sha256=hashlib.sha256(plan_file.read_bytes()).hexdigest(),offers=offers,train_objective_parent=risk,train_objective_candidate=risk-chosen[1],whole_plane_selected=chosen[2],whole_plane_extra_bytes=chosen[0],domain='19 non-layer17 down projections; gate/up and the 61 experts restored in layer 17 stay fixed',embedding_bytes_saved=saving),indent=2))
