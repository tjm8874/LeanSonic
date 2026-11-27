"""Export reference margins and candidate scores for the three-chunk comparison."""
import csv
from contextlib import ExitStack
import struct
import sys
from pathlib import Path
import numpy as np

oracle, *scores = map(Path, sys.argv[1:])
assert len(scores) == 3
chunks = [('test', 36), ('test', 98), ('valid', 85)]
out = csv.writer(sys.stdout, lineterminator='\n')
out.writerow(['split','chunk','position','target','bf16_top','bf16_runner_up','bf16_logprob_gap',
              '571_top','571_kl','925_top','925_kl','927_top','927_kl'])
with oracle.open('rb') as ref, ExitStack() as stack:
    magic, ctx, vocab, count = struct.unpack('<8sIII',ref.read(20))
    assert (magic, ctx, vocab, count) == (b'_logits_',512,200058,3)
    ids=np.fromfile(ref,dtype='<u4',count=ctx*count).reshape(count,ctx)
    streams=[csv.reader(stack.enter_context(f.open())) for f in scores]
    for i,(split,original_chunk) in enumerate(chunks):
        for pos in range(ctx//2,ctx-1):
            scale,offset=struct.unpack('<ff',ref.read(8))
            codes=np.fromfile(ref,dtype='<u2',count=vocab)
            first=int(codes.argmax());highest=int(codes[first]);codes[first]=0
            second=int(codes.argmax());target=int(ids[i,pos+1])
            result=[split,original_chunk,pos,target,first,second,scale*(highest-int(codes[second]))]
            for stream in streams:
                row=next(stream)
                assert list(map(int,row[:3]))==[i,pos,target]
                assert (int(row[3])==first)==(row[6]=='true')
                result.extend([int(row[3]),row[4]])
            out.writerow(result)
    assert ref.read(1)==b'' and all(next(stream,None) is None for stream in streams)
