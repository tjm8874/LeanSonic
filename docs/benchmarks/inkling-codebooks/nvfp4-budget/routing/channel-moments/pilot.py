"""Sample gate/up ternary fits using the original weight errors and train-only diagonal input moments."""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

momfile, official = map(Path, sys.argv[1:3])
mom = np.memmap(momfile, dtype='<f8', mode='r', shape=(20, 128, 2304))
with official.open('rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))

def scale(code):
    return np.left_shift(8 + (code & 7), (code >> 3) - 1)

def array(key, shape):
    h = hdr[key]
    assert h['shape'] == list(shape)
    lo, hi = h['data_offsets']
    assert hi - lo == np.prod(shape)
    return np.memmap(official, dtype='u1', mode='r', offset=n + 8 + lo, shape=shape)

rng = np.random.default_rng(8874)
results = []
for il in range(20):
    name = f'model.llm.layers.{il}.mlp.experts.w13_weight'
    packed = array(name, (128, 2304, 1152)).reshape(128, 2304, 144, 8)
    sc = array(name+'_scale', (128, 2304, 144))
    exp = np.arange(128)[:, None]
    rows = rng.integers(0, 2304, (128, 64))
    blocks = rng.integers(0, 144, (128, 64))
    lanes = blocks[:, :, None]*16 + np.arange(16)
    old = sc[exp, rows, blocks].astype('i8')
    assert old.min()>=64 and old.max()<=126
    chars = packed[exp, rows, blocks]
    codes = np.stack((chars & 7, (chars>>4)&7),axis=-1).reshape(128,64,16)
    # This objective only needs magnitudes, since each retained ternary sign follows its source.
    raw = np.array([0,1,2,3,4,6,8,12])[codes]*scale(old)[:,:,None]
    # Each expert/channel gets half uniform prior, half normalized W146 train energy.
    energy = mom[il]
    avg = energy.mean(axis=1,keepdims=True)
    importance = .5 + .5*np.divide(energy,avg,out=np.ones_like(energy),where=avg>0)
    weight = importance[exp[:,:,None],lanes]
    best = [np.full(old.shape,np.inf) for _ in range(2)]
    codes_best = [np.zeros_like(old) for _ in range(2)]
    loss_best = [np.zeros_like(raw) for _ in range(2)]
    # Matches the original bound and its (absolute distance, lower code) tie order.
    for offset in [0,-1,1,-2,2,3,4,5,6]:
        c = (old+offset).clip(64,126)
        level = 12*scale(c)[:,:,None]
        err = np.minimum(raw**2,(raw-level)**2)
        for arm,score in enumerate([err.sum(axis=-1),(err*weight).sum(axis=-1)]):
            take = score < best[arm]
            best[arm][take] = score[take]
            codes_best[arm][take] = c[take]
            loss_best[arm][take] = err[take]
    after = (loss_best[1]*weight).sum(axis=-1)
    before = (loss_best[0]*weight).sum(axis=-1)
    assert np.all(after<=before) and np.all(loss_best[1].sum(axis=-1)>=loss_best[0].sum(axis=-1))
    results.append(dict(layer=il,blocks=int(old.size),changed=int((codes_best[0]!=codes_best[1]).sum()),
                        weighted_error_prior=float(before.sum()),weighted_error_trial=float(after.sum()),
                        ordinary_error_prior=int(loss_best[0].sum()),ordinary_error_trial=int(loss_best[1].sum())))
out=dict(source_sha256=hashlib.sha256(momfile.read_bytes()).hexdigest(),seed=8874,blocks_per_expert=64,layers=results)
for metric in ['weighted_error','ordinary_error']:
    out[metric+'_pct']=100*(sum(r[metric+'_trial'] for r in results)/sum(r[metric+'_prior'] for r in results)-1)
out['changed_blocks']=sum(r['changed'] for r in results)
Path('pilot.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['changed_blocks'],out['weighted_error_pct'],out['ordinary_error_pct'])
