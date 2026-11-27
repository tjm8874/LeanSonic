"""Compare every shipped byte to the prescribed source, independently of the assembler."""
import hashlib
import json
import struct
import sys
from pathlib import Path

def index(model):
    path=model/"model-00001-of-00001.safetensors"
    with path.open("rb") as f:
        n,=struct.unpack("<Q",f.read(8))
        return path,8+n,json.loads(f.read(n))

def digest(f,pos,n):
    f.seek(pos)
    h=hashlib.sha256()
    while n:
        b=f.read(min(8<<20,n));assert b
        h.update(b);n-=len(b)
    return h.digest()

low,high,last,receipt=map(Path,sys.argv[1:])
r=json.loads(receipt.read_text())
planes=set(r["selected"])
assert len(planes)==len(r["selected"])
lp,lo,lm=index(low);hp,ho,hm=index(high);fp,fo,fm=index(last)
assert set(lm)==set(hm)==set(fm)
from_control=from_parent=0
with lp.open("rb") as lf,hp.open("rb") as hf,fp.open("rb") as final:
    for name,v in fm.items():
        if name=="__metadata__":
            assert lm[name]==hm[name]==v
            continue
        restored=name.removesuffix(".nvscale_four_v1") in planes or name.removesuffix("_scale.mean8_v1") in planes
        source,offset,meta=(hf,ho,hm[name]) if restored else (lf,lo,lm[name])
        a,b=v["data_offsets"];s,t=meta["data_offsets"]
        assert v["dtype"]==meta["dtype"] and v["shape"]==meta["shape"] and b-a==t-s
        assert digest(final,fo+a,b-a)==digest(source,offset+s,t-s)
        if restored:from_control+=1
        else:from_parent+=1
assert from_parent+from_control==999 and from_control==len(planes)*2
assert len({(d/"tiktoken/tokenizer.model").read_bytes() for d in (low,high,last)})==1
assert sum(f.stat().st_size for f in last.rglob("*") if f.is_file())==r["package_bytes"]
print(json.dumps(dict(unchanged_parent_payloads=from_parent,exact_control_payloads=from_control,
                      package_bytes=r["package_bytes"],restored=sorted(planes)),indent=2))
