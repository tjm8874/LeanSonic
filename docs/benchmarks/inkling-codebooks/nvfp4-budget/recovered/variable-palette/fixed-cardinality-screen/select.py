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
    observations={'0124':ref['sample']}
    for palette in ('0136','h24','h135','0126'):
        os.environ['INKLING_OFFLINE_PALETTE']=palette
        path=Path('sample-'+palette+'.json')
        measured=observe(official,selection,routing,probe)
        write(path,measured)
        observations[palette]=measured
    # Match the previous fit's constraint and byte accounting for an unbiased comparison.
    unit=1<<20
    risk,choices=fit(observations,ref,math.ceil((base-ref['target'])/unit))
    names=sorted(ref['modes'])
    modes=dict(zip(names,(m for _,m in choices),strict=True))
    palettes=dict(zip(names,(p for p,_ in choices),strict=True))
    size=base+sum(128*(observations[p][name][m]['sample_bytes']-ref['sample'][name]['control']['sample_bytes']) for name,(p,m) in zip(names,choices,strict=True))
    assert size<=ref['target'] and risk<=ref['risk']
    write(Path('plan.json'),dict(target=ref['target'],projected=size,risk=risk,
         prior_risk=ref['risk'],modes=modes,palettes=palettes,
         parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
         selection_sha256=ref['selection_sha256'],routing_sha256=ref['routing_sha256']))
