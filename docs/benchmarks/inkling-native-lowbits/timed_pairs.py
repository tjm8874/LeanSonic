"""Summarize the matched five-cycle comparisons without treating requests as independent."""
import csv
import itertools
import json
import statistics
from pathlib import Path
import sys

d=Path(sys.argv[1]);fixture=Path(sys.argv[2])
ids=list(map(int,fixture.read_text().strip().split(',')))
r=json.loads((d/'receipt.json').read_text())
with (d/'requests.csv').open() as f: data=list(csv.DictReader(f))
out={}
for pair in ('iq3','iq1'):
    rows=[x for x in data if x['pair']==pair]
    assert len(rows)==100
    table={}
    for field in ('prefill_ms','decode32_ms'):
        means={}
        for cycle in range(1,6):
            order=['lean','llama','llama','lean'] if cycle%2 else ['llama','lean','lean','llama']
            for leg,arm in enumerate(order):
                legrows=[v for v in rows if int(v['cycle'])==cycle and int(v['leg'])==leg]
                assert len(legrows)==5 and all(x['runtime']==arm for x in legrows)
            for arm in ('lean','llama'):
                vals=[float(x[field]) for x in rows if x['runtime']==arm and int(x['cycle'])==cycle and int(x['run'])>0]
                assert len(vals)==8
                means[cycle,arm]=statistics.mean(vals)
        def ratio(indices):return sum(means[c,'llama'] for c in indices)/sum(means[c,'lean'] for c in indices)
        dist=sorted(ratio(ix) for ix in itertools.product(range(1,6),repeat=5))
        table[field]=dict(lean_speedup=ratio(range(1,6)),paired_bootstrap_95=[dist[78],dist[-79]])
    for arm in ('lean','llama'):
        trace=r['traces'][str(r['models'][pair]['timing'][arm]['trace'])]
        assert len(trace)==len(ids)==32
        table[arm+'_predictions_matching_forced_inputs']=sum(x==y for x,y in zip(ids,trace))
    out[pair]=table
(d/'timing-pairs.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
