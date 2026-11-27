from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast,json,struct,subprocess,sys
p=Path(sys.argv[1])/'model-00001-of-00001.safetensors'
with p.open('rb') as f:
 n,=struct.unpack('<Q',f.read(8));h=json.loads(f.read(n));base=f.tell()
names=sorted(k for k in h if k.endswith('_scale.mean8_v1') and '.mlp.experts.' in k)
assert len(names)==40
def inspect(k):
 w=144 if '.w13_' in k else 72
 a,b=h[k]['data_offsets']
 text=subprocess.check_output(['./scales',str(p),str(base+a),str(b-a),str(w),'dump' if k=='model.llm.layers.0.mlp.experts.w13_weight_scale.mean8_v1' else 'verify'],text=True)
 out={}
 for line in text.splitlines():
  group,storage,baseline,counts=line.split(' ',3);storage,baseline=map(int,(storage,baseline));counts=ast.literal_eval(counts)
  assert sum(counts)*int(group)==128*2304*w and len(counts)==7 and baseline==128*2304*w*6//8+4
  out[group]={'encoded':storage,'baseline':baseline,'width_counts':counts}
 return k,out
with ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(inspect,names))
groups=['128','256','512','1024','4096','expert'];summary={}
for k in groups:
 a=b=n=0
 for plane,v in results.items():
  t=v[k if k!='expert' else str(2304*(144 if '.w13_' in plane else 72))]
  a+=min(t['encoded'],t['baseline']);b+=t['baseline'];n+=t['encoded']<t['baseline']
 summary[k]={'selected_bytes':a,'baseline_bytes':b,'profitable_planes':n,'savings':b-a}
Path('receipt.json').write_text(json.dumps({'planes':results,'summary':summary},indent=2)+'\n');print(summary)
