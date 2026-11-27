from concurrent.futures import ThreadPoolExecutor
import json,struct,subprocess,sys
from pathlib import Path
folder,rank,dest=map(Path,sys.argv[1:]);dest.mkdir()
shard=folder/'model-00001-of-00001.safetensors'
with shard.open('rb') as f:
 n,=struct.unpack('<Q',f.read(8));h=json.loads(f.read(n));base=f.tell()
selection=json.loads(rank.read_text())['selected']
def evaluate(item):
 name,ids=item;args=[];key=name+'.nvscale_four_v1'
 for k in (key,name+'_scale.mean8_v1'):
  lo,hi=h[k]['data_offsets'];args.extend((str(base+lo),str(hi-lo)))
 width=144 if 'w13_weight' in name else 72
 output=dest/(key+'.frame')
 raw=subprocess.check_output(['./banks',str(shard),*args,str(width),','.join(map(str,ids)),str(output)],text=True)
 lines=raw.splitlines();_,choice,size,_,decoded=lines[-1].split()
 trials={k:int(v) for k,v in (line.split() for line in lines[:-1])}
 assert int(size)==min(trials.values())==trials[choice] and int(decoded)==128*2304*width*8
 assert output.is_file()==(choice!='original')
 if output.is_file():assert output.stat().st_size==int(size)
 v=dict(choice=choice,bytes=int(size),decoded_bytes=int(decoded),trials=trials)
 (dest/(key+'.json')).write_text(json.dumps(v)+'\n')
 return key,v
with ThreadPoolExecutor(max_workers=2) as pool:r=dict(pool.map(evaluate,sorted(selection.items())))
Path('fit.json').write_text(json.dumps(r,indent=2)+'\n')
