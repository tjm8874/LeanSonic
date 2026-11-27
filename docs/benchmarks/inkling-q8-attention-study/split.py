"""Isolate the output projection from the other attention matrices."""
from pathlib import Path
import json, struct, sys, subprocess, os
baseline, candidate, output = (Path(x).resolve() for x in sys.argv[1:4])
shard = 'model-00001-of-00001.safetensors'
def header(source):
 with (source / shard).open('rb') as f:
  n, = struct.unpack('<Q', f.read(8))
  return json.loads(f.read(n)), 8 + n
stock, offset = header(baseline)
trial, other_offset = header(candidate)
assert stock == trial and offset == other_offset
for arm in ('wo_only', 'except_wo'):
 dest = output / arm
 dest.mkdir()
 tmp = dest / 'pending'
 subprocess.run(['cp', '--reflink=auto', baseline/shard, tmp], check=True)
 assert not os.path.samefile(tmp, baseline/shard)
 tensors = 0
 with tmp.open('r+b') as out, (candidate/shard).open('rb') as src:
  for name, item in stock.items():
   if '.attn.' not in name or not name.endswith(('.q8_64.codes','.q8_64.scales')): continue
   if ('.attn.wo_ud.weight.' in name) != (arm == 'wo_only'):continue
   a,b = item['data_offsets']
   src.seek(a + offset); out.seek(a + offset); data = src.read(b-a)
   assert len(data) == b-a
   out.write(data); tensors += 1
 assert tensors == (48 if arm == 'wo_only' else 304)
 tmp.rename(dest/shard)
 (dest/'tiktoken').symlink_to((baseline/'tiktoken').resolve())
