"""Count real grouped weight tiles and bound inactive MMA lanes from actual routing."""
import collections,json,sys
from pathlib import Path
rows=[]
for line in Path(sys.argv[1]).read_text().splitlines():
    _,layer,kind,counts=line.split(' ',3)
    counts=json.loads(counts)
    assert len(counts)==128 and sum(counts)==8192 and min(counts)>=0
    rows.append((int(layer),kind,counts))
assert len(rows)==4*48 and all(rows[i][0:2]==rows[i%48][0:2] for i in range(len(rows)))
reports=[]
for i,label in enumerate(('readme','test-0','test-35','valid-84')):
    visits=rows[i*48:(i+1)*48]
    counts=[n for _,kind,by_expert in visits if kind=='nvfp4' for n in by_expert]
    assert counts
    mma_slots=sum((n+7)//8*8 for n in counts)
    active=sum(counts)
    hist=collections.Counter(counts)
    reports.append(dict(fixture=label,nvfp4_visits=len(counts)//128,routed_rows=active,
        observed_count_histogram=sorted(hist.items()),per_expert_max=max(counts),round_to_8_slots=mma_slots,
        fraction_unused_in_issued_8_row_groups=(mma_slots-active)/mma_slots,
        tiles={t:dict(live_weight_tiles=sum((n+t-1)//t for n in counts),
                      nominal_token_capacity=t*sum((n+t-1)//t for n in counts)) for t in (64,128,160)}))
print(json.dumps(reports,indent=2))
