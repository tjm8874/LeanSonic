"""Verify every DSpark output element independently of the Rust packer."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


class Shard:
    def __init__(self,path):
        self.path=path
        with path.open('rb') as f:
            n,=struct.unpack('<Q',f.read(8))
            self.start=n+8
            self.header=json.loads(f.read(n))
        self.metadata=self.header.pop('__metadata__',{})
        cursor=0
        for name,entry in sorted(self.header.items(),key=lambda item:item[1]['data_offsets']):
            lo,hi=entry['data_offsets']
            size={'I8':1,'BF16':2,'F32':4}[entry['dtype']]
            assert lo==cursor and hi-lo==int(np.prod(entry['shape']))*size,name
            cursor=hi
        assert self.start+cursor==path.stat().st_size

    def tensor(self,name,dtype):
        entry=self.header[name]
        assert entry['dtype']==dtype
        dt={'I8':'i1','BF16':'<u2','F32':'<f4'}[dtype]
        return np.memmap(self.path,dtype=dt,mode='r',offset=self.start+entry['data_offsets'][0],shape=tuple(entry['shape']))


def digest(path):
    with path.open('rb') as fp:
        return hashlib.file_digest(fp,'sha256').hexdigest()


def verify(source,target):
    src,dst=Shard(source),Shard(target)
    assert src.metadata==dst.metadata
    transformed=set()
    consumed=set()
    matrices={}
    def check(prefix,names,jctx=False):
        codes=dst.tensor(prefix+'.q8_64.codes','I8')
        scales=dst.tensor(prefix+'.q8_64.scales','F32')
        transformed.update((prefix+'.q8_64.codes',prefix+'.q8_64.scales'))
        row=0
        for name in names:
            orig=src.tensor(name,'BF16')
            consumed.add(name)
            for start in range(0,len(orig),64):
                raw=orig[start:start+64]
                if jctx:
                    j01=dst.tensor('markov_head.rnn_proj.j01.weight','BF16')
                    np.testing.assert_array_equal(j01[start:start+len(raw)],raw[:,:512])
                    raw=raw[:,512:]
                x=(raw.astype('<u4')<<16).view('<f4').reshape(-1,64)
                assert np.isfinite(x).all()
                a=np.abs(x).max(axis=1,keepdims=True)
                expected=np.float32(1/127)*a
                inverse=np.divide(np.float32(127),a,out=np.zeros_like(a),where=a!=0)
                values=x*inverse
                rounded=np.copysign(np.floor(np.abs(values).astype('f8')+0.5),values)
                q=np.clip(rounded,-127,127).astype('i1').reshape(raw.shape)
                stop=row+len(raw)
                np.testing.assert_array_equal(codes[row:stop],q)
                np.testing.assert_array_equal(scales[row:stop].view('<u4'),expected.reshape(len(raw),-1).view('<u4'))
                row=stop
        assert row==len(codes) and scales.shape==(row,codes.shape[1]//64)
        matrices[prefix]=dict(rows=row,cols=codes.shape[1],codes=int(codes.size),scales=int(scales.size))

    for i in range(3):
        layer=f'layers.{i}.'
        for prefix,parts in (
                ('self_attn.qkv_proj',['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj']),
                ('self_attn.kv_ctx_proj',['self_attn.k_proj_ctx','self_attn.v_proj_ctx']),
                ('self_attn.o_proj',['self_attn.o_proj']),('mlp.gate_up_proj',['mlp.gate_proj','mlp.up_proj']),
                ('mlp.down_proj',['mlp.down_proj'])):
            check(layer+prefix,[layer+k+'.weight' for k in parts])
    check('markov_head.rnn_proj.jctx',['markov_head.rnn_proj.weight'],True)
    transformed.add('markov_head.rnn_proj.j01.weight')
    others=src.header.keys()-consumed
    assert set(dst.header)==transformed|others
    for name in sorted(others):
        np.testing.assert_array_equal(src.tensor(name,'BF16'),dst.tensor(name,'BF16'))
    return dict(source_bytes=source.stat().st_size,source_sha256=digest(source),packed_bytes=target.stat().st_size,
                packed_sha256=digest(target),verified_output_tensors=len(dst.header),retained_bf16=sorted(others),verified_matrices=matrices)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('original',type=Path);parser.add_argument('packed',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.original,args.packed),indent=2))
