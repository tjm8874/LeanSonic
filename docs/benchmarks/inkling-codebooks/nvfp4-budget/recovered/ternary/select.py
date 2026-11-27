import hashlib
import json
import math
import os
import sys
from pathlib import Path
from observe import observe

def write(path, data):
    with path.open('x') as f:
        json.dump(data,f,indent=2)

def fit(observations, reference, required):
    scores={0:(0.,())}
    for name in sorted(reference['modes']):
        choices=[('0124','control')]
        choices.extend((palette,mode) for palette in observations for mode in ('all','low'))
        base=observations['0124'][name]['control']
        assert all(o[name]['control']==base for o in observations.values())
        next_scores={}
        for palette,mode in choices:
            row=observations[palette][name][mode]
            units=(128*(base['sample_bytes']-row['sample_bytes']))//(1<<20)
            for saved,(risk,previous) in scores.items():
                bucket=min(required,saved+units)
                state=(risk+row['weighted_cost'],previous+((palette,mode),))
                if bucket not in next_scores or state<next_scores[bucket]:
                    next_scores[bucket]=state
        scores=next_scores
    return scores[required]

if __name__=='__main__':
    official,selection,routing,probe,parent=map(Path,sys.argv[1:])
    base=13479710294
    ref=json.loads(parent.read_text())
    assert ref['selection_sha256']==hashlib.sha256(selection.read_bytes()).hexdigest()
    assert ref['routing_sha256']==hashlib.sha256(routing.read_bytes()).hexdigest()
    previous,reference_target=Path(os.environ['PALETTE_PREVIOUS']),7764291776
    target=reference_target-(16<<20)
    observations={'0124':ref['sample']}
    for name in ('024','012','0h124','01234','01246','012346'):
        observations[name]=json.loads((previous/('sample-'+name+'.json')).read_bytes())
    for name in ('01','02','03','04','06'):
        os.environ['INKLING_OFFLINE_PALETTE']=name
        rows=observe(official,selection,routing,probe)
        write(Path('sample-'+name+'.json'),rows)
        observations[name]=rows
    names=sorted(ref['modes'])
    minimum=base+sum(min(128*(rows[k][m]['sample_bytes']-ref['sample'][k]['control']['sample_bytes'])
                          for rows in observations.values() for m in ('control','low','all')) for k in names)
    summary=dict(reference_gguf_bytes=reference_target,target=target,lowest_projected=minimum,
        uniform_full_plane_estimates={label:base+sum(128*(rows[k]['all']['sample_bytes']-ref['sample'][k]['control']['sample_bytes']) for k in names) for label,rows in observations.items()})
    write(Path('feasibility.json'),summary)
    unit=1<<20
    required=math.ceil((base-target)/unit)
    conservative=sum(max((128*(ref['sample'][k]['control']['sample_bytes']-rows[k][m]['sample_bytes']))//unit
                    for rows in observations.values() for m in ('control','low','all')) for k in names)
    if conservative<required:
        print('No feasible plan at this budget among the measured options',flush=True)
        sys.exit(0)
    risk,choices=fit(observations,ref,required)
    modes=dict(zip(names,(m for _,m in choices),strict=True))
    palettes=dict(zip(names,(p for p,_ in choices),strict=True))
    size=base+sum(128*(observations[p][name][m]['sample_bytes']-ref['sample'][name]['control']['sample_bytes']) for name,(p,m) in zip(names,choices,strict=True))
    assert size<=target
    write(Path('plan.json'),dict(target=target,projected=size,risk=risk,modes=modes,palettes=palettes,
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),selection_sha256=ref['selection_sha256'],routing_sha256=ref['routing_sha256']))
