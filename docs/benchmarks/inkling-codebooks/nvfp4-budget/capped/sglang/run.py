import json,os,re,subprocess as sp,sys
from pathlib import Path
cycle=int(sys.argv[1])
model=os.environ['LEAN_MODEL']
arms=['lean','sglang','sglang','lean'] if cycle%2 else ['sglang','lean','lean','sglang']
exe=os.environ['REPLAY']
for leg,arm in enumerate(arms):
    tag=Path('runs')/f'{cycle}-{leg}-{arm}';tag.parent.mkdir(exist_ok=True)
    env={**os.environ,'RESULT':str(tag.with_suffix('.metrics'))}
    cmd=[exe,str(model),'prompt.ids','continuation.ids'] if arm=='lean' else [sys.executable,'sgl.py']
    with tag.with_suffix('.out').open('x') as stdout,tag.with_suffix('.log').open('x') as stderr:
        p=sp.Popen(cmd,stdout=stdout,stderr=stderr,env=env)
        pidfile=tag.with_suffix('.pid');pidfile.write_text(str(p.pid))
        peak=sp.Popen([sys.executable,'peak.py',pidfile,tag.with_suffix('.peak')])
        code=p.wait();sampler=peak.wait()
        assert code==sampler==0,(str(tag),code,sampler)
    v=json.loads(tag.with_suffix('.peak').read_text())
    assert v['peak_bytes']>0 and v['positive_polls']>100
    if arm=='lean':
        raw=tag.with_suffix('.out').read_text()
        phases=re.findall(r'^phases ([\d.]+) ([\d.]+)$',raw,re.M)
        tokens=list(map(int,re.findall(r'^T (\d+)$',raw,re.M)))
        assert len(phases)==8 and len(tokens)==8*33
        rows=[dict(request=i,prefill_ms=float(p),decode32_ms=float(d),ids=tokens[33*i:33*(i+1)]) for i,(p,d) in enumerate(phases)]
    else: rows=json.loads(tag.with_suffix('.metrics').read_text())
    assert all(row['ids']==rows[0]['ids'] for row in rows)
    dest=tag.with_suffix('.json')
    dest.write_text(json.dumps(dict(arm=arm,cycle=cycle,leg=leg,peak=v,requests=rows),indent=2)+'\n')
