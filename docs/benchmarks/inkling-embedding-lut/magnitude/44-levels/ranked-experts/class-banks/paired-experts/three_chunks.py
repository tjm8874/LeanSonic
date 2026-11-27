from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import struct,sys
test,valid,out=map(Path,sys.argv[1:]);selection=[('test',36),('test',98),('valid',85)]
with ExitStack() as stack:
    sources={}
    for name,path in [('test',test),('valid',valid)]:
        original=path.open('rb')
        f=stack.enter_context(original);magic,nctx,nv,n=struct.unpack('<8sIII',f.read(20))
        assert (magic,nctx,nv,n)==(b'_logits_',512,200058,100)
        sources[name]=f
    row_bytes=(8+2*(nv+nv%2))*(nctx-1-nctx//2)
    with out.open('xb') as output:
        output.write(struct.pack('<8sIII',magic,nctx,nv,len(selection)))
        for split,chunk in selection:
            f=sources[split];f.seek(20+chunk*nctx*4);output.write(f.read(nctx*4))
        for split,chunk in selection:
            f=sources[split];f.seek(20+n*nctx*4+chunk*row_bytes)
            for offset in range(0,row_bytes,1<<20):
                count=min(1<<20,row_bytes-offset);data=f.read(count);assert len(data)==count;output.write(data)
