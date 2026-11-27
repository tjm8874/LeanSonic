"""Account for every byte in an original Inkling GGUF or a LeanSonic directory."""
import argparse
import ctypes
import json
import math
from pathlib import Path
import struct

def gguf(path, libpath):
    lib = ctypes.CDLL(str(libpath))
    lib.ggml_type_name.argtypes = [ctypes.c_int]
    lib.ggml_type_name.restype = ctypes.c_char_p
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_row_size.restype = ctypes.c_size_t
    with path.open('rb') as f:
        def number(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        def string():
            return f.read(number('<Q')).decode()
        def skip(kind,n=1):
            widths={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
            if kind in widths:
                f.seek(widths[kind]*n,1)
            elif kind==8:
                for _ in range(n):
                    f.seek(number('<Q'),1)
            elif kind==9:
                for _ in range(n):
                    item,count=struct.unpack('<IQ',f.read(12))
                    skip(item,count)
            else:
                raise ValueError(kind)
        assert f.read(4)==b'GGUF' and number('<I')==3
        n,m=struct.unpack('<QQ',f.read(16))
        alignment=32
        for _ in range(m):
            key,kind=string(),number('<I')
            if key=='general.alignment':
                assert kind==4
                alignment=number('<I')
            else:
                skip(kind)
        entries=[]
        for _ in range(n):
            key=string()
            dims=[number('<Q') for _ in range(number('<I'))]
            kind,offset=struct.unpack('<IQ',f.read(12))
            size=lib.ggml_row_size(kind,dims[0])*math.prod(dims[1:])
            entries.append(dict(name=key,shape=dims,format=lib.ggml_type_name(kind).decode(),offset=offset,bytes=size))
        header=(f.tell()+alignment-1)//alignment*alignment
    size=path.stat().st_size
    rows=sorted(entries,key=lambda e:e['offset'])
    assert rows[0]['offset']==0
    for i,row in enumerate(rows):
        end=rows[i+1]['offset'] if i+1<len(rows) else size-header
        gap=end-row['offset']-row['bytes']
        assert 0<=gap<alignment,(row,gap)
    return dict(format='GGUF',file=path.name,bytes=size,header_bytes=header,padding_bytes=size-header-sum(x['bytes'] for x in rows),tensors=rows)

def shard(directory):
    path=directory/'model-00001-of-00001.safetensors'
    with path.open('rb') as f:
        size,=struct.unpack('<Q',f.read(8))
        h=json.loads(f.read(size))
    rows=[]
    for k,v in h.items():
        if k=='__metadata__':continue
        begin,end=v['data_offsets']
        rows.append(dict(name=k,shape=v['shape'],format=v['dtype'],offset=begin,bytes=end-begin))
    rows.sort(key=lambda e:e['offset'])
    assert rows[0]['offset']==0
    assert all(a['offset']+a['bytes']==b['offset'] for a,b in zip(rows,rows[1:]))
    assert 8+size+sum(r['bytes'] for r in rows)==path.stat().st_size
    side={str(p.relative_to(directory)):p.stat().st_size for p in directory.rglob('*') if p.is_file() and p!=path}
    assert set(side)=={'tiktoken/tokenizer.model'}
    return dict(format='safetensors',bytes=path.stat().st_size+sum(side.values()),header_bytes=8+size,sidecars=side,tensors=rows)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('model',type=Path)
    ap.add_argument('--ggml',type=Path)
    a=ap.parse_args()
    print(json.dumps(shard(a.model) if a.model.is_dir() else gguf(a.model,a.ggml),separators=(',',':')))
