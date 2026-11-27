"""Copy unmodified reference rows and token histories for three maxima and their predecessors."""
import hashlib,json,struct,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:])
chunks=(0,1,76,77,89,90)
with src.open('rb') as r,dst.open('xb') as w:
 hdr=r.read(20);assert hdr[:8]==b'_logits_'
 ctx,vocab,n=struct.unpack_from('<III',hdr,8);assert (ctx,vocab)==(512,200058) and n>=100
 w.write(hdr[:16]+struct.pack('<I',len(chunks)))
 for c in chunks:
  r.seek(20+4*ctx*c);ids=r.read(4*ctx);assert len(ids)==4*ctx;w.write(ids)
 # The vocab is even here. Each stored probability row has two floats followed by uint16 codes.
 span=(ctx-1-ctx//2)*(8+2*vocab)
 assert src.stat().st_size==20+n*ctx*4+n*span
 for c in chunks:
  r.seek(20+n*ctx*4+c*span)
  for off in range(0,span,1<<20):
   amount=min(1<<20,span-off);buf=r.read(amount);assert len(buf)==amount;w.write(buf)
with dst.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(original_chunks=chunks,bytes=dst.stat().st_size,subset_sha256=h)))
