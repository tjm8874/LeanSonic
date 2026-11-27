"""Wall-clock occupancy from the union of GPU intervals. PDL kernels can overlap."""
import collections
import itertools
import json
import sqlite3
import sys
from pathlib import Path

db=sqlite3.connect(f'file:{Path(sys.argv[1]).resolve()}?mode=ro',uri=True)
raw=db.execute("SELECT k.start,k.end,n.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds n ON k.demangledName=n.id")
events=collections.defaultdict(collections.Counter)
for start,end,name in raw:
    assert end>start
    events[start][name]+=1
    events[end][name]-=1
live=collections.Counter()
durations=collections.Counter()
solo=collections.Counter()
pairs=collections.Counter()
last=min(events)
for t,delta in sorted(events.items()):
    time=t-last;last=t
    depth=sum(live.values())
    durations[depth]+=time
    if depth==1:solo[next(iter(live))]+=time
    if len(live)>1:
        for pair in itertools.combinations(sorted(live),2):pairs[pair]+=time
    live.update(delta)
    assert all(x>=0 for x in live.values())
    live=+live
assert not live
span=max(events)-min(events)
assert sum(durations.values())==span
print(json.dumps(dict(span_ms=span/1e6,depth_ms={k:v/1e6 for k,v in sorted(durations.items())},
    sole_active_kernel_ms={k:v/1e6 for k,v in solo.most_common()},
    overlap_ms_top_pairs=[dict(kernels=k,ms=v/1e6) for k,v in pairs.most_common(20)]),indent=2))
