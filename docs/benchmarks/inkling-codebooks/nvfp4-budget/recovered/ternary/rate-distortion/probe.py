import json,os,subprocess as sp,sys
from pathlib import Path
from observe import observe
official,selection,routing,probe=map(Path,sys.argv[1:])
out={}
for penalty in [-1,0,4,12,24,48]:
 os.environ['TERNARY_NONZERO_COST']=str(penalty)
 path=Path(f'sample-{penalty}.json')
 if not path.exists():
  rows=observe(official,selection,routing,probe);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(rows));tmp.rename(path)
 out[penalty]=json.loads(path.read_bytes());source=out[-1]
 projected=7734665510+sum(128*(r['all']['sample_bytes']-source[k]['all']['sample_bytes']) for k,r in out[penalty].items())
 risk=sum(r['all']['weighted_cost'] for r in out[penalty].values())
 print(penalty,projected,risk,flush=True)
