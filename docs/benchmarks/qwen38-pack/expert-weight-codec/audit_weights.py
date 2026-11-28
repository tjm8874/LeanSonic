"""Reread every newly packed weight record and all remaining tensors against the source."""
import argparse
import hashlib
import json
import mmap
import struct
from pathlib import Path

import zstandard

from pack_weights import COUNT, MAGIC, RECORD_BYTES, SUFFIX, WEIGHT


def header(buf):
    size, = struct.unpack_from("<Q", buf)
    end = size + 8
    if size > 32 << 20 or end > len(buf):
        raise ValueError("invalid header")
    data = json.loads(buf[8:end])
    data.pop("__metadata__", None)
    return data, end


def check(src, packed):
    index = "model.safetensors.index.json"
    a,b = [json.loads((d/index).read_text())["weight_map"] for d in (src,packed)]
    normalized = {key + SUFFIX if WEIGHT.fullmatch(key) else key: shard for key,shard in a.items()}
    if normalized != b:
        raise ValueError("inconsistent index")
    aux = {p.name for p in src.iterdir()} - set(a.values()) - {index}
    if set(p.name for p in packed.iterdir()) != set(a.values()) | aux | {index}:
        raise ValueError("package file list changed")
    for name in aux:
        if (src/name).read_bytes() != (packed/name).read_bytes():
            raise ValueError(f"changed auxiliary file {name}")
    digest=hashlib.sha256()
    plain = count = 0
    decoder=zstandard.ZstdDecompressor()
    for shard in sorted(set(a.values())):
        with (src/shard).open("rb") as f, (packed/shard).open("rb") as g, \
            mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as before, mmap.mmap(g.fileno(),0,access=mmap.ACCESS_READ) as after:
            old, start = header(before)
            new, dst = header(after)
            for key in sorted(old):
                coded = bool(WEIGHT.fullmatch(key))
                nk = key + SUFFIX if coded else key
                if a.get(key) != shard or b.get(nk) != shard:
                    raise ValueError(f"unindexed tensor {key}")
                t, u = old[key], new[nk]
                lo, hi = (start+v for v in t["data_offsets"])
                left, right = (dst+v for v in u["data_offsets"])
                if not coded:
                    if (t["dtype"], t["shape"], hi-lo) != (u["dtype"], u["shape"], right-left):
                        raise ValueError(f"changed tensor format {key}")
                    for off in range(0,hi-lo,4<<20):
                        n=min(4<<20,hi-lo-off)
                        if before[lo+off:lo+off+n] != after[left+off:left+off+n]:
                            raise ValueError(f"changed tensor {key}")
                    plain+=1
                    continue
                if (u["dtype"],u["shape"]) != ("U8",[right-left]) or hi-lo != COUNT*RECORD_BYTES:
                    raise ValueError(f"bad weight plane {key}")
                if struct.unpack_from("<8sII",after,left) != (MAGIC,COUNT,RECORD_BYTES):
                    raise ValueError("bad frame header")
                offsets=struct.unpack_from(f"<{COUNT+1}I",after,left+16)
                payload=left+16+4*(COUNT+1)
                if offsets[0] or payload+offsets[-1]!=right:
                    raise ValueError("bad payload extent")
                for i, (x,y) in enumerate(zip(offsets,offsets[1:])):
                    if not 0<y-x<=RECORD_BYTES:
                        raise ValueError("bad record length")
                    data=after[payload+x:payload+y]
                    if len(data)!=RECORD_BYTES:
                        data=decoder.decompress(data,max_output_size=RECORD_BYTES)
                    if before[lo+i*RECORD_BYTES:lo+(i+1)*RECORD_BYTES]!=data:
                        raise ValueError(f"weight mismatch {key} {i}")
                    digest.update(data)
                    count+=1
    if count!=144*COUNT:
        raise ValueError("unexpected record count")
    return {"records":count,"raw_routed_weight_bytes":count*RECORD_BYTES,"ordinary_tensors":plain,"auxiliary_files":sorted(aux),
            "routed_weight_sha256_shard_key_order":digest.hexdigest(),
            "source_bytes":sum(p.stat().st_size for p in src.iterdir()),"package_bytes":sum(p.stat().st_size for p in packed.iterdir())}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("source","packed"):
        ap.add_argument(arg,type=Path)
    args=ap.parse_args()
    print(json.dumps(check(args.source,args.packed),indent=2))
