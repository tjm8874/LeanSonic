"""Build a physical two-table sign/magnitude BF16 frame from a frozen fit."""
import hashlib,json,struct,subprocess,sys
from pathlib import Path
import numpy as np
shard,mapping,packer,dest=map(Path,sys.argv[1:]);dest.mkdir()
with shard.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];t=json.loads(f.read(n))['model.llm.embed.weight']
words=np.memmap(shard,'<u2','r',offset=8+n+t['data_offsets'][0],shape=tuple(t['shape']))
m=np.fromfile(mapping,'<u2');assert m.size==32768
hist=np.zeros(32768,np.int64)
for i in range(0,len(words),1024):hist+=np.bincount(m[words[i:i+1024]&32767].ravel(),minlength=32768)
codes=np.flatnonzero(hist);table=codes[np.argsort(-hist[codes],kind='stable')].astype('<u2')
assert 255<len(table)<=511
rank=np.full(32768,65535,np.uint16);rank[table]=np.arange(len(table))
output_hash=hashlib.sha256();counter=0
with (dest/'sign').open('xb') as signs,(dest/'first').open('xb') as first,(dest/'extra').open('xb') as extra:
 for i in range(0,len(words),256):
    token=words[i:i+256].ravel();quant=m[token&32767];index=rank[quant]
    assert index.max()<len(table) and token.size%8==0
    primary=np.minimum(index,255).astype(np.uint8);secondary=(index[index>=255]-255).astype(np.uint8)
    sign=np.packbits((token>>15).astype(np.uint8),bitorder='little')
    signs.write(sign);first.write(primary);extra.write(secondary);counter+=secondary.size
    # Rebuild independently from rank bytes and check rounded values and sign bits.
    rebuilt=primary.astype(np.uint16);rebuilt[primary==255]+=secondary
    decoded=table[rebuilt]|(np.unpackbits(sign,bitorder='little').astype('<u2')<<15)
    assert np.array_equal(decoded,quant|(token&32768));output_hash.update(decoded)
for name in ('first','extra'):
 subprocess.run([str(packer),'pack',str(dest/name),str(dest/(name+'.rans'))],check=True)
sizes=[(dest/(name+'.rans')).stat().st_size for name in ('first','extra')]
with (dest/'embedding.frame').open('xb') as f:
 f.write(struct.pack('<IIII',len(table),counter,*sizes));f.write(table.tobytes())
 for name in ('sign','first.rans','extra.rans'):
    with (dest/name).open('rb') as g:
        while buf:=g.read(1<<20):f.write(buf)
with (dest/'embedding.frame').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(values=words.size,representatives=len(table),secondary_codes=counter,frame_bytes=(dest/'embedding.frame').stat().st_size,frame_sha256=digest,decode_sha256=output_hash.hexdigest(),rans_bytes=sizes),indent=2))
