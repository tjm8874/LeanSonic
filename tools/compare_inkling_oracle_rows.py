"""Compare two encoded BF16 oracles against exactly the same engine logits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

def read_header(stream):
    magic, ctx, vocab, chunks = struct.unpack('<8sIII', stream.read(20))
    if (magic, ctx, vocab) != (b'_logits_', 512, 200058) or chunks < 1:
        raise ValueError("expected an Inkling WikiText KLD reference")
    return ctx, vocab, chunks

def reference_row(stream, offset, vocab):
    stream.seek(offset)
    scale, minimum = struct.unpack('<ff', stream.read(8))
    codes = np.frombuffer(stream.read(2*vocab), dtype='<u2')
    return np.float32(scale) * codes.astype(np.float32) + np.float32(minimum)

def compare(logits, reference):
    top = int(logits.argmax())
    lse = logits[top] + np.float32(np.log(np.exp(logits-logits[top]).sum(dtype=np.float64)))
    active = reference > np.float32(-16)
    divergence = np.sum(np.exp(reference[active]) * (reference[active] - logits[active] + lse), dtype=np.float64)
    return dict(kl=float(divergence), reference_top=int(reference.argmax()), engine_top=top)

def run(args):
    selected = {}
    for item in args.row:
        chunk, pos = map(int,item.split('/'))
        if chunk < 0 or not 256 <= pos < 511:
            raise ValueError(f"invalid scored position {item}")
        selected.setdefault(chunk,set()).add(pos)
    results = []
    with args.canonical.open('rb') as canonical, args.serial.open('rb') as serial:
        ctx, vocab, chunks = read_header(canonical)
        if read_header(serial) != (ctx, vocab, chunks) or max(selected) >= chunks:
            raise ValueError("references must cover the same chunks and selected rows")
        token_bytes = canonical.read(chunks*ctx*4)
        if len(token_bytes) != chunks * ctx * 4 or token_bytes != serial.read(len(token_bytes)):
            raise ValueError("reference input token IDs differ")
        tokens = np.frombuffer(token_bytes, dtype='<u4').reshape(chunks,ctx)
        row_start = 20 + len(token_bytes)
        row_size = 8 + 2*vocab
        for chunk, positions in sorted(selected.items()):
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                prompt, raw = folder/'prompt.ids', folder/'rows.f32'
                ids = tokens[chunk]
                prompt.write_text(','.join(map(str,ids[:257]))+'\n')
                forced = ','.join(map(str,ids[257:max(positions)+1]))
                env = dict(os.environ, LEANSONIC_EVAL_FORCE=forced, LEANSONIC_EVAL_DUMP=str(raw), LEANSONIC_KV='fp16')
                subprocess.run([args.cli.resolve(),'bench','--weights',args.weights,'--tokens','@'+str(prompt),'--gen','0','--spec','0','--repeat','1'],env=env,check=True,stdout=subprocess.DEVNULL)
                logits = np.memmap(raw, dtype='<f4', mode='r').reshape(-1,vocab)
                if len(logits) != max(positions) - 255 or not np.isfinite(logits).all():
                    raise ValueError("invalid engine logits dump")
                for pos in sorted(positions):
                    row = logits[pos-256]
                    offset = row_start + (chunk*255 + pos-256)*row_size
                    results.append(dict(chunk=chunk, position=pos, target=int(ids[pos+1]), engine_logits_sha256=hashlib.sha256(row).hexdigest(), canonical=compare(row,reference_row(canonical,offset,vocab)), serial=compare(row,reference_row(serial,offset,vocab))))
                del logits
    print(json.dumps(results,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cli','weights','canonical','serial'):
        parser.add_argument('--'+name,type=Path, required=True)
    parser.add_argument('--row',action='append',required=True)
    run(parser.parse_args())
