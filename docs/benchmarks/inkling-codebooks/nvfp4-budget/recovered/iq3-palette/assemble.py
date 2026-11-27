import argparse, hashlib, json, shutil, struct, subprocess, tempfile
from pathlib import Path

def header(path):
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        return length+8,json.loads(f.read(length))

def copy_slice(src,dest,size):
    digest=hashlib.sha256()
    while size:
        block=src.read(min(size,8<<20))
        if not block:raise EOFError('incomplete tensor')
        digest.update(block)
        if dest is not None:dest.write(block)
        size-=len(block)
    return digest.hexdigest()

def run(official, compact, selection, plan, encoder, dest):
    shard='model-00001-of-00001.safetensors'
    srcfile=compact/shard; ob,orig=header(official); cb,meta=header(srcfile)
    cuts=json.loads(selection.read_text())['cutoffs']
    modes=json.loads(plan.read_text())['modes']
    assert set(modes)==set(cuts) and len(cuts)==40
    dest.mkdir()
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        work=Path(tmp);stats={};frames={}
        for name,limits in sorted(cuts.items()):
            mode=modes[name]
            if mode=='control': continue
            weight,scale=name+'.nvscale_direct_v1',name+'_scale.mean8_v1'
            assert weight in meta and scale in meta
            oa,oz=orig[name]['data_offsets'];sc=orig[name+'_scale']
            sa,sz=sc['data_offsets'];assert oz-oa==8*(sz-sa)
            tmpcut=work/'limits';tmpcut.write_bytes(bytes(x if x>=0 else 255 for x in limits))
            wp,spth=work/weight,work/scale
            vals=subprocess.check_output([encoder,official,str(ob+oa),str(ob+sa),str(sc['shape'][-1]),tmpcut,mode,wp,spth],text=True)
            ws,ss,changed=map(int,vals.split());assert ws==wp.stat().st_size and ss==spth.stat().st_size
            frames[weight]=wp;frames[scale]=spth
            stats[name]=dict(mode=mode,weight_bytes=ws,scale_bytes=ss,reduced_blocks=changed)
        end=0;updated={};digests={}
        for key,data in meta.items():
            if key=='__metadata__':updated[key]=data;continue
            lo,hi=data['data_offsets'];n=frames[key].stat().st_size if key in frames else hi-lo
            updated[key]={**data,'data_offsets':[end,end+n]}
            if key in frames:updated[key]['shape']=[n]
            end+=n
        hdr=json.dumps(updated,separators=(',',':'),ensure_ascii=False).encode();hdr+=b' '*(-len(hdr)%8)
        with (dest/shard).open('xb') as output,srcfile.open('rb') as source:
            output.write(struct.pack('<Q',len(hdr)));output.write(hdr)
            for key,data in meta.items():
                if key=='__metadata__': continue
                if key in frames:
                    with frames[key].open('rb') as f:digests[key]=copy_slice(f,output,frames[key].stat().st_size)
                else:
                    a,b=data['data_offsets'];source.seek(cb+a);digests[key]=copy_slice(source,output,b-a)
        shutil.copytree(compact/'tiktoken', dest/'tiktoken')
        db,check=header(dest/shard)
        assert check==updated and db+end==(dest/shard).stat().st_size
        with (dest/shard).open('rb') as f:
            for key,h in digests.items():
                a,b=check[key]['data_offsets'];f.seek(db+a);assert copy_slice(f,None,b-a)==h
        files={}
        for f in sorted(dest.rglob('*')):
            if f.is_file() and work not in f.parents:
                with f.open('rb') as inputs:files[str(f.relative_to(dest))]=dict(bytes=f.stat().st_size,sha256=hashlib.file_digest(inputs,'sha256').hexdigest())
        assert set(files)=={shard,'tiktoken/tokenizer.model'}
        print(json.dumps(dict(package_bytes=sum(v['bytes'] for v in files.values()),files=files,planes=stats,verified_tensors=len(digests)),indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('official','compact','selection','plan','encoder','dest'):parser.add_argument(name,type=Path)
    args=parser.parse_args();run(**vars(args))
