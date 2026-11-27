import json
import struct
import subprocess
import sys
from pathlib import Path

def parse(file):
    n, = struct.unpack('<Q',file.read(8))
    return 8+n,json.loads(file.read(n))

ternary,official,extractor=map(Path,sys.argv[1:])
key='model.llm.layers.0.mlp.experts.w13_weight'
with ternary.open('rb') as f:
    base,meta=parse(f)
offsets=[]
for suffix in ('.nvscale_direct_v1','_scale.mean8_v1'):
    start,end=meta[key+suffix]['data_offsets']
    offsets.extend([str(base+start),str(end-start)])
subprocess.run([extractor,ternary,*offsets,'144'],check=True)
with official.open('rb') as f:
    base,meta=parse(f)
    entry=meta[key+'_amax']
    lo,hi=entry['data_offsets']
    assert entry['dtype']=='F32' and hi-lo==512
    f.seek(base+lo)
    with Path('amax.bin').open('xb') as dst:
        dst.write(f.read(512))
