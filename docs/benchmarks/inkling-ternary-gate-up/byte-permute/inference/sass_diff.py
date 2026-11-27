"""Compare disassembled instruction words, including control bits, by CUDA function name."""
from pathlib import Path
import hashlib,json,re,sys
def functions(path):
    source=Path(path).read_text()
    parts=re.split(r"^\s*Function\s*:\s*([^\s]+)\s*$",source,flags=re.M)
    out={}
    for name,body in zip(parts[1::2],parts[2::2]):
        words=re.findall(r"/\*\s*0x([0-9a-fA-F]{16})\s*\*/",body)
        assert words and len(words)%2==0 and name not in out,name
        out[name]=(len(words)//2,hashlib.sha256(''.join(words).encode()).hexdigest())
    assert out
    return out
a,b=map(functions,sys.argv[1:])
assert a.keys()==b.keys()
changed={k:dict(prior=a[k],trial=b[k]) for k in a if a[k]!=b[k]}
decode={k:a[k] for k in a if 'ternary_dp4a' in k}
assert decode
print(json.dumps(dict(compared=len(a),changed=changed,ternary_decode={k:dict(prior=v,trial=b[k],identical=v==b[k]) for k,v in decode.items()}),indent=2))
