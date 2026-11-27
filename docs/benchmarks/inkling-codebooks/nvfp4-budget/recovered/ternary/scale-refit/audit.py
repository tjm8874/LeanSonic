import json,subprocess,sys
from pathlib import Path
from tempfile import TemporaryDirectory
from assemble import header
def verify(official,parent,model,selection,binary):
    shard='model-00001-of-00001.safetensors'
    off,src=header(official);bc,old=header(parent/shard);nc,new=header(model/shard)
    result={}
    with TemporaryDirectory() as work:
        cutfile=Path(work)/'limits'
        for name,cut in sorted(json.loads(selection.read_text())['cutoffs'].items()):
            cutfile.write_bytes(bytes(x if x>=0 else 255 for x in cut))
            key=name+'_scale.mean8_v1';w=name+'.nvscale_direct_v1'
            blo,bhi=old[key]['data_offsets'];nlo,nhi=new[key]['data_offsets'];wlo,whi=new[w]['data_offsets']
            a=off+src[name]['data_offsets'][0];s=src[name+'_scale'];b=off+s['data_offsets'][0]
            args=[binary,official,str(a),str(b),parent/shard,str(bc+blo),str(bhi-blo),model/shard,str(nc+nlo),str(nhi-nlo),str(nc+wlo),str(whi-wlo),str(s['shape'][-1]),cutfile]
            n,c,z=map(int,subprocess.check_output(args,text=True).split());assert n==4096 and c+z<=n
            result[name]=dict(sampled=n,changed=c,all_zero=z)
    assert len(result)==40
    return result
if __name__=='__main__':print(json.dumps(verify(*map(Path,sys.argv[1:])),indent=2))
