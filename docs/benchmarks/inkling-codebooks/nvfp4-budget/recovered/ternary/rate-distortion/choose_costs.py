import json,math
from pathlib import Path
levels=[-1,0,4,12,24,48];tbl={l:json.loads(Path(f'sample-{l}.json').read_bytes()) for l in levels}
# Bound a plane's savings downwards before using it in the knapsack.
unit=1<<20; parent_bytes=7734665510; target=7718950382-(16<<20); states={0:(0.,())}
keys=sorted(tbl[-1])
for key in keys:
 choices={}
 for cost in levels:
  r=tbl[cost][key]['all'];savings=128*(tbl[-1][key]['all']['sample_bytes']-r['sample_bytes'])
  for s,(risk,seq) in states.items():
   k=s+savings//unit;v=(risk+r['weighted_cost'],seq+(cost,))
   if k not in choices or v<choices[k]:choices[k]=v
 states=choices
risk,settings=min(v for bytes_saved,v in states.items() if bytes_saved>=math.ceil((parent_bytes-target)/unit))
pen=dict(zip(keys,settings,strict=True))
projected=parent_bytes+sum(128*(tbl[p][k]['all']['sample_bytes']-tbl[-1][k]['all']['sample_bytes']) for k,p in pen.items())
prior=sum(tbl[-1][k]['all']['weighted_cost'] for k in keys)
assert projected<=target and risk<prior
obj=dict(prior_bytes=parent_bytes,target=target,projected=projected,weighted_risk=risk,prior_weighted_risk=prior,penalties=pen,modes=dict.fromkeys(keys,'all'),palettes=dict.fromkeys(keys,'06'))
Path('plan.json').write_text(json.dumps(obj,indent=2)+'\n')
print(projected,risk,prior,sorted((k,settings.count(k)) for k in set(settings)))
