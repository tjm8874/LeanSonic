"""Test a native Q4_K embedding as an independent lower-capacity control."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

shard,gguf,refit,libpath,index=map(Path,sys.argv[1:])
inv=json.loads(index.read_text())
tensor=next(t for t in inv['tensors'] if t['name']=='token_embd.weight')
assert tensor['format']=='q4_K' and tensor['shape']==[2304,201024] and gguf.stat().st_size==inv['bytes']
with shard.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(n))['model.llm.embed.weight']
assert header['dtype']=='BF16' and header['shape']==[201024,2304]
base=np.memmap(shard,'<u2','r',offset=8+n+header['data_offsets'][0],shape=tuple(header['shape']))
trial=np.memmap(refit,'<u2','r',shape=base.shape)
packed=np.memmap(gguf,'u1','r',offset=inv['header_bytes']+tensor['offset'],shape=(base.shape[0],9*144))
lib=ct.CDLL(str(libpath))
decode=lib.dequantize_row_q4_K
decode.restype=None;decode.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int64]
stats={k:dict(energy=0.,sse=0.,exact=0,max_error=0.) for k in ('native','refit')}
rels={k:[] for k in stats}
hashes=[hashlib.sha256(),hashlib.sha256()]
def f32(bf):return (bf.astype(np.uint32)<<16).view(np.float32)
for i in range(0,len(base),128):
    original=base[i:i+128];payload=packed[i:i+128]
    tmp=np.empty(original.shape,dtype='<f4');decode(payload.ctypes.data,tmp.ctypes.data,tmp.size)
    assert np.isfinite(tmp).all()
    bits=tmp.view(np.uint32);bf=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
    hashes[0].update(payload);hashes[1].update(bf)
    target=f32(original).astype(np.float64);energy=(target*target).sum(axis=1)
    for label,approx in [('native',bf),('refit',trial[i:i+128])]:
        delta=target-f32(approx);sse=(delta*delta).sum(axis=1);s=stats[label]
        s['sse']+=float(sse.sum());s['energy']+=float(energy.sum())
        s['exact']+=int((original==approx).sum())
        s['max_error']=max(s['max_error'],float(np.abs(delta).max()))
        valid=energy>0
        rels[label].extend(np.sqrt(sse[valid]/energy[valid]).tolist())
for key,s in stats.items():
    s['relative_rmse']=(s.pop('sse')/s.pop('energy'))**.5
    s['nonzero_row_relative_rmse_p50_p95_p99']=np.percentile(rels[key],[50,95,99]).tolist()
print(json.dumps(dict(values=base.size,native_tensor=tensor,native_payload_sha256=hashes[0].hexdigest(),native_decoded_bf16_sha256=hashes[1].hexdigest(),scores=stats),indent=2))
