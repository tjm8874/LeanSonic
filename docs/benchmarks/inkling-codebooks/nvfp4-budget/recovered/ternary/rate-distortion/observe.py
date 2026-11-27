import hashlib,json,math,struct,subprocess as sp,sys, tempfile
from pathlib import Path
from assemble import header
MODES=('control','low','all')
def observe(official,selection,routing,probe):
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
    return observed
