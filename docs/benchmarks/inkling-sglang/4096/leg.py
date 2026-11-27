import csv,json,os,subprocess,sys
from pathlib import Path
tag,arm=sys.argv[1:];p=Path(tag)
env=dict(os.environ,RESULT=str(p.with_suffix('.metrics')))
cmd=[os.environ['SGL_PYTHON'],'sgl.py'] if arm=='sglang' else ['./bench',os.environ['LEAN_MODEL']]
with p.with_suffix('.out').open('x') as out,p.with_suffix('.log').open('x') as err:
    child=subprocess.Popen(cmd,stdout=out,stderr=err,env=env)
    pid=p.with_suffix('.pid');pid.write_text(str(child.pid))
    monitor=subprocess.Popen([sys.executable,'peak.py',str(pid),str(p.with_suffix('.peak'))])
    rc=child.wait(); sr=monitor.wait()
    assert rc==sr==0,(tag,rc,sr)
peak=json.loads(p.with_suffix('.peak').read_text());assert peak['positive_polls']>100
if arm=='sglang':
    rows=json.loads(p.with_suffix('.metrics').read_text())
    assert [r['rep'] for r in rows]==list(range(8)) and len({tuple(r['ids']) for r in rows})==1
    anchor=','.join(map(str,rows[0]['ids'][:32]))+'\n'
    if tag=='anchor':Path('anchor.ids').write_text(anchor)
    else:assert Path('anchor.ids').read_text()==anchor
else:
    rows=[dict(rep=int(rep),mode=mode,prefill_ms=float(pre),decode_ms=float(dec),ids=list(map(int,ids.split())))
          for rep,mode,pre,dec,ids in csv.reader(p.with_suffix('.out').read_text().splitlines())]
    assert len(rows)==16 and all(len(r['ids'])==33 for r in rows)
    for mode in ['forced','greedy']:
        group=[r for r in rows if r['mode']==mode]
        assert [r['rep'] for r in group]==list(range(8)) and len({tuple(r['ids']) for r in group})==1
p.with_suffix('.json').write_text(json.dumps(dict(arm=arm,peak=peak,rows=rows))+'\n')
