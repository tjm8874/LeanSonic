"""Estimate cross-entropy on independent routed expert blocks using BF16 group parameters."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from screen import BLOCK, EXPERT, Shards, bf16, unpack


def classify(sc_bytes, bi_bytes, projection):
    sc = bf16(sc_bytes)
    bi = bf16(bi_bytes)
    if not (np.isfinite(sc).all() and np.isfinite(bi).all()):
        raise ValueError("nonfinite affine parameters")
    # Which fractional Q4 code would encode zero? Bin at half-code spacing.
    zero = np.rint(np.divide(-2 * bi, sc, out=np.zeros_like(sc), where=sc != 0)).clip(0,30).astype(np.int32)
    exp = ((np.frombuffer(sc_bytes, dtype="<u2") >> 7) & 255).astype(np.int32)
    mag = np.clip(exp - 116, 0, 7)
    return ((projection * 31 + zero) * 8 + mag).astype("<u2")


def samples(directory):
    shards = Shards(directory)
    keys = sorted(k for k in shards.tensors if EXPERT.fullmatch(k))
    if len(keys) != 144:
        raise ValueError("expected 144 expert planes")
    for key in keys:
        _, proj = EXPERT.fullmatch(key).groups()
        proj_id = ("gate","up","down").index(proj)
        blocks = shards.tensors[key][2] // BLOCK
        for label, idx in zip(("train","test","test"),(blocks//7,blocks//2,5*blocks//6),strict=True):
            off = idx*BLOCK
            raw = shards.read(key,off,BLOCK,"U32")
            group = classify(*(shards.read(key.removesuffix("weight")+name,off//8,BLOCK//8,"BF16") for name in ("scales","biases")),proj_id)
            yield label, group, raw


def table(hist):
    # Integer frequencies with all symbols reachable and a power-of-two total.
    prob=(hist+1)/(hist.sum(axis=1,keepdims=True)+16)
    scaled=prob*(4096-16)
    freq=1+np.floor(scaled).astype(np.int32)
    for h in range(len(freq)):
        delta=4096-freq[h].sum()
        order=np.argsort(-(scaled[h]-np.floor(scaled[h])),kind="stable")
        freq[h,order[:delta]]+=1
    assert np.all(freq.sum(axis=1)==4096) and np.min(freq)>0
    return freq


def main(model, destination):
    destination.mkdir()
    count={"train":np.zeros((3*31*8,16),dtype=np.int64),"test":np.zeros((3*31*8,16),dtype=np.int64)}
    with (destination/"test.bin").open("xb") as stream:
        for label, cls, raw in samples(model):
            codes=unpack(raw).reshape(-1,32)
            flat=(cls.astype(np.int32)[:,None]*16+codes).ravel()
            count[label]+=np.bincount(flat,minlength=count[label].size).reshape(count[label].shape)
            if label=="test":
                stream.write(cls.tobytes())
                stream.write(raw)
    freq=table(count["train"])
    freq.astype("<u2").tofile(destination/"frequencies.u16")
    train=count["train"];test=count["test"]
    metric=lambda obs,ft: float(np.sum(obs*(-np.log2(ft/4096)))/obs.sum())
    print(json.dumps({"train_weights":int(train.sum()),"test_weights":int(test.sum()),
                      "table_bytes":freq.size*2, "train_bits_per_weight":metric(train,freq),
                      "heldout_bits_per_weight":metric(test,freq),
                      "unconditional_heldout_bits":metric(test.sum(axis=0,keepdims=True),table(train.sum(axis=0,keepdims=True)))},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",type=Path);p.add_argument("destination",type=Path)
    a=p.parse_args();main(a.model,a.destination)
