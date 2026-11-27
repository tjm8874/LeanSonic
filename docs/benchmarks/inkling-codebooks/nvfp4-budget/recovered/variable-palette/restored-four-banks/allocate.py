"""Trade actual four-bank frame bytes for whole control planes at the fixed IQ3_XXS budget."""
import hashlib
import json
import math
import shutil
import struct
import sys
from contextlib import ExitStack
from pathlib import Path

SHARD = "model-00001-of-00001.safetensors"
CAP = 11663309504

def header(folder):
    with (folder / SHARD).open("rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        return 8 + n, json.loads(f.read(n))

def copy(src, dst, size):
    h = hashlib.sha256()
    while size:
        chunk = src.read(min(size, 8 << 20))
        assert chunk
        if dst is not None:
            dst.write(chunk)
        h.update(chunk)
        size -= len(chunk)
    return h.hexdigest()

def full_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

def pair(meta, stem):
    weights = [stem+s for s in (".nvscale_four_v1",".nvscale_direct_v1") if stem+s in meta]
    assert len(weights)==1
    return weights[0], stem+"_scale.mean8_v1"

def nbytes(meta, key):
    a,b=meta[key]["data_offsets"]
    return b-a

def choose(offers, available):
    # Each frontier retains the largest training gain at a byte cost. A cheaper,
    # at-least-as-good subset dominates any costlier state for every later choice.
    states = {0:(0.,())}
    for name, price, gain in offers:
        assert price >= 0 and gain > 0
        entries = dict(states)
        for old,(score,picks) in states.items():
            cost=old+price
            if cost > available: continue
            trial=(score+gain,picks+(name,))
            if cost not in entries or trial[0] > entries[cost][0]: entries[cost]=trial
        best=-1.
        states={}
        for cost,value in sorted(entries.items()):
            if value[0]>best:
                states[cost]=value
                best=value[0]
    cost=max(states)
    return cost,*states[cost]

def run(base,ctrl,planfile,samples,original,out):
    po,pm=header(base);co,cm=header(ctrl)
    assert pm.get("__metadata__")==cm.get("__metadata__")
    plan=json.loads(planfile.read_text())
    stats={"0124":json.loads(original.read_text())["sample"]}
    for pal in sorted(set(plan["palettes"].values())-{"0124"}):
        stats[pal]=json.loads((samples/f"sample-{pal}.json").read_text())
    prices={};gains={};cost0=0.
    for stem,mode in plan["modes"].items():
        row=stats[plan["palettes"][stem]][stem]
        cost0+=row[mode]["weighted_cost"]
        gains[stem]=row[mode]["weighted_cost"]-row["control"]["weighted_cost"]
        prices[stem]=sum(nbytes(cm,k) for k in pair(cm,stem))-sum(nbytes(pm,k) for k in pair(pm,stem))
    assert len(gains)==40 and math.isclose(cost0,plan["risk"],rel_tol=1e-10)
    offers=sorted((k,prices[k],gains[k]) for k in gains if gains[k]>0)
    assert all(cost>=0 for _,cost,_ in offers)
    basebytes=full_size(base)
    cost,gain,selected=choose(offers,CAP-basebytes-4096)
    assert selected and gain>0
    replacements={}
    for stem in selected:
        replacements.update(zip(pair(pm,stem),pair(cm,stem),strict=True))
    new={};sources={};end=0
    for k,entry in pm.items():
        if k=="__metadata__":new[k]=entry;continue
        source,key,offset,meta=(1,replacements[k],co,cm) if k in replacements else (0,k,po,pm)
        assert key not in new
        a,b=meta[key]["data_offsets"]; size=b-a
        new[key]=dict(meta[key],data_offsets=[end,end+size]);end+=size
        sources[key]=(source,offset+a,size)
    assert len(sources)==999
    raw=json.dumps(new,separators=(",",":"),ensure_ascii=False).encode()
    raw+=b" "*(-len(raw)%8)
    out.mkdir()
    with ExitStack() as stack:
        files=[stack.enter_context((folder/SHARD).open("rb")) for folder in (base,ctrl)]
        hashes={}
        with (out/SHARD).open("xb") as f:
            f.write(struct.pack("<Q",len(raw)));f.write(raw)
            for key,(source,offset,size) in sources.items():
                files[source].seek(offset)
                hashes[key]=copy(files[source],f,size)
    shutil.copytree(base/"tiktoken",out/"tiktoken")
    bo,bm=header(out)
    assert bm==new
    with (out/SHARD).open("rb") as f:
        for k,h in hashes.items():
            a,b=new[k]["data_offsets"];f.seek(bo+a)
            assert copy(f,None,b-a)==h
    files={}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            with path.open("rb") as f:
                files[str(path.relative_to(out))]=dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,"sha256").hexdigest())
    assert (base/"tiktoken/tokenizer.model").read_bytes()==(out/"tiktoken/tokenizer.model").read_bytes()
    assert basebytes<full_size(out)<=CAP
    return dict(cap=CAP,package_bytes=full_size(out),parent_bytes=basebytes,frame_bytes_added=cost,
                train_objective_parent=cost0,train_objective_candidate=cost0-gain,selected=selected,
                offers=[dict(plane=k,extra_bytes=p,training_gain=g) for k,p,g in offers],files=files,
                changed_tensors=len(replacements),verified_tensors=len(hashes),
                plan_sha256=hashlib.sha256(planfile.read_bytes()).hexdigest())

if __name__=="__main__": print(json.dumps(run(*map(Path,sys.argv[1:])),indent=2))
