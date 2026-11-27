import heapq,json,collections
from pathlib import Path
import numpy as np
w=np.memmap('official.bf16',dtype='<u2',mode='r');counts={}
for mode in ['exp','mag','frac']:
 h=np.zeros(256,dtype=np.int64)
 for i in range(0,len(w),1<<21):
    v=w[i:i+(1<<21)];vs= ((v>>7)&255) if mode=='exp' else (v>>8)&127 if mode=='mag' else ((v&127)|((v>>8)&128))
    h+=np.bincount(vs,minlength=256)
 counts[mode]=list(map(int,h))
Path('huffman.json').write_text(json.dumps(counts))
for mode,h in counts.items():
 if mode=='frac':continue
 print(mode,sorted(enumerate(h),key=lambda r:-r[1])[:16]);n=sum(h);raw_bits=8 if mode=='exp' else 9;escaped_bits=8 if mode=='exp' else 7
 for k in (4,7,8,10,12,15,20):
    freq=sorted(h,reverse=True)[:k];q=[(x,str(i)) for i,x in enumerate(freq+[n-sum(freq)])];heapq.heapify(q)
    codes={i:'' for i in range(k+1)};cost=0
    while len(q)>1:
        a,x=heapq.heappop(q);b,y=heapq.heappop(q);cost+=a+b
        for ix in x.split(','):codes[int(ix)]='0'+codes[int(ix)]
        for ix in y.split(','):codes[int(ix)]='1'+codes[int(ix)]
        heapq.heappush(q,(a+b,x+','+y))
    print(k,round((n*raw_bits+cost+(n-sum(freq))*escaped_bits)/8),list(map(len,codes.values())))
