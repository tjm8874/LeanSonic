"""Exercise the measured-width encoder on special values and sparse/full alphabets."""
import hashlib,importlib.util,json,struct,subprocess,sys
from pathlib import Path
import numpy as np
here=Path.cwd()
path,packer,unpack=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('encoder',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def digest(b):return hashlib.sha256(b).hexdigest()
results=[]
for case,alphabet in [('special',np.array([0,32768,0x7fc1,0xffc1,0x7f80,0xff80,0x3f80],dtype='<u2')),
                      ('all',np.arange(65536,dtype='<u2'))]:
    d=here/case;d.mkdir()
    rng=np.random.default_rng(392)
    if case=='all':
        words=rng.permutation(alphabet).reshape(-1,32)
    else:
        words=rng.choice(alphabet,size=(256,32))
    raw=words.astype('<u2').tobytes()
    header=json.dumps({'model.llm.embed.weight':dict(dtype='BF16',shape=words.shape,data_offsets=[0,len(raw)])}).encode()
    header+=b' '*(-len(header)%8)
    shard=d/'shard';shard.write_bytes(struct.pack('<Q',len(header))+header+raw)
    auto=module.encode(shard,packer,d/'auto')
    assert auto['decoded_sha256']==digest(raw)
    sizes={}
    for k_str,expected in auto['measured_frame_bytes'].items():
        k=int(k_str);out=d/str(k)
        got=module.encode(shard,packer,out,k)
        assert got['frame_bytes']==expected
        frame=(out/'embedding.frame').read_bytes()
        magic,n,bits,upper=struct.unpack_from('<4sIII',frame)
        assert magic==b'BRK1' and bits==k
        table=np.frombuffer(frame,dtype='<u2',count=n,offset=16)
        a=16+2*n;lo=np.frombuffer(frame[a:-upper],dtype='u1').reshape(-1,k)
        (d/'high.rans').write_bytes(frame[-upper:])
        subprocess.run([unpack,d/'high.rans',str(words.size),d/'high.raw'],check=True)
        hi=np.fromfile(d/'high.raw',dtype='u1').astype('<u2')<<k
        lanes=np.zeros(len(lo),dtype='<u8')
        for j in range(k):lanes|=lo[:,j].astype('<u8')<<(8*j)
        for j in range(8): hi[j::8]|=((lanes>>(k*j))&((1<<k)-1)).astype('<u2')
        assert table[hi].tobytes()==raw
        sizes[k]=len(frame)
        if k==auto['low_bits']:assert frame==(d/'auto/embedding.frame').read_bytes()
    assert min((v,k) for k,v in sizes.items())==(auto['frame_bytes'],auto['low_bits'])
    results.append(dict(case=case,values=words.size,widths=sizes,selected=auto['low_bits'],decoded_sha256=digest(raw)))
print(json.dumps(results,indent=2))
