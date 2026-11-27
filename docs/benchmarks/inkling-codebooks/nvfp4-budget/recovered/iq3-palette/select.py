import hashlib,json,math,struct,subprocess as sp,sys, tempfile
from pathlib import Path
from assemble import header
MODES=('control','low','all')
def select(official,selection,routing,base_size,probe,target):
    base,meta=header(official);cuts=json.loads(selection.read_text())['cutoffs'];factors=json.loads(routing.read_text())['factor']
    observed={}
    assert len(factors)==20 and all(len(layer)==128 for layer in factors)
    with official.open('rb') as f,tempfile.TemporaryDirectory() as tmp:
        for seed,(k,limit) in enumerate(sorted(cuts.items()),933):
            layer=int(k.split('.')[3]);ss=meta[k+'_scale'];wo,we=meta[k]['data_offsets'];so,se=ss['data_offsets'];width=ss['shape'][-1]
            assert (we-wo)//8==se-so==128*2304*width
            aa,az=meta[k+'_amax']['data_offsets'];f.seek(base+aa);amp=struct.unpack('<128f',f.read(az-aa));assert all(math.isfinite(x) and x>0 for x in amp)
            lut=Path(tmp)/'cuts';lut.write_bytes(bytes(x if x>=0 else 255 for x in limit))
            rows=sp.check_output([probe,official,str(base+wo),str(base+so),str(width),lut,str(seed)],text=True).splitlines()
            assert len(rows)==131
            errors=[0.,0.,0.];energy=0.
            for expert,line in enumerate(rows[:128]):
                name,idx,signal,*loss=line.split();assert (name,int(idx))==('expert',expert)
                coef=amp[expert]**2
                energy+=int(signal)*coef
                for i,n in enumerate(loss):errors[i]+=int(n)*coef*factors[layer][expert]
            options={}
            for i,line in enumerate(rows[128:]):
                mode,w,sc,n,eligible,loss=line.split();assert mode==MODES[i]
                options[mode]=dict(sample_bytes=int(w)+int(sc),sample_blocks=int(n),eligible=int(eligible),weighted_cost=errors[i]/energy*(3 if 6<=layer<18 else 1)*int(n))
            observed[k]=options
    # Minimize training-weighted normalized distortion with a conservative byte constraint.
    # Round each projected saving down to whole MiB so the discretized plan meets target.
    unit=1<<20;need=math.ceil((base_size-target)/unit)
    scores={0:(0.,())}
    for options in observed.values():
        nxt={};base=options['control']['sample_bytes']
        for saved,(risk,path) in scores.items():
            for mode,v in options.items():
                diff=128*(base-v['sample_bytes']);bucket=min(need,saved+diff//unit)
                state=(risk+v['weighted_cost'],path+(mode,))
                if bucket not in nxt or state<nxt[bucket]:nxt[bucket]=state
        scores=nxt
    risk,path=scores[need];modes=dict(zip(observed,path,strict=True))
    projected=base_size+sum(128*(o[modes[k]]['sample_bytes']-o['control']['sample_bytes']) for k,o in observed.items())
    assert projected<=target
    return dict(target=target,projected=projected,risk=risk,modes=modes,sample=observed,
                selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),routing_sha256=hashlib.sha256(routing.read_bytes()).hexdigest())
if __name__=='__main__':
    off,sel,routing,exe,out=map(Path,sys.argv[1:]);r=select(off,sel,routing,13479710294,exe,11663309504-32*1024*1024)
    with out.open('x') as f:json.dump(r,f,indent=2)
