import csv,json,os,subprocess,sys
from pathlib import Path
tag,arm=sys.argv[1:];p=Path(tag)
env=dict(os.environ,VLLM_LEG_RESULT=str(p.with_suffix('.metrics')))
cmd=[os.environ['VLLM_PYTHON'],'native.py'] if arm=='vllm' else (['./bench',os.environ['LEAN_MODEL']] if arm=='lean' else ['./llama',os.environ['GGUF']])
if arm=='llama':env['LD_LIBRARY_PATH']=os.environ['NATIVE_LIB']+':'+env.get('LD_LIBRARY_PATH','')
with p.with_suffix('.out').open('x') as out,p.with_suffix('.log').open('x') as err:
    child=subprocess.Popen(cmd,stdout=out,stderr=err,env=env)
    pid=p.with_suffix('.pid');pid.write_text(str(child.pid))
    monitor=subprocess.Popen([sys.executable,'peak.py',str(pid),str(p.with_suffix('.peak'))])
    rc=child.wait(); sr=monitor.wait()
    assert rc==sr==0,(tag,rc,sr)
peak=json.loads(p.with_suffix('.peak').read_text());assert peak['positive_polls']>100
if arm=='vllm':
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
if arm=='llama':assert 'offloaded 25/25 layers' in p.with_suffix('.log').read_text()
if arm=='lean':
    log=p.with_suffix('.log').read_text();assert '24/24 (Spilled: 0)' in log and 'KV: fp16' in log and 'Spec: off' in log and 'Packed 48 expert scale planes' in log
p.with_suffix('.json').write_text(json.dumps(dict(arm=arm,peak=peak,rows=rows))+'\n')
