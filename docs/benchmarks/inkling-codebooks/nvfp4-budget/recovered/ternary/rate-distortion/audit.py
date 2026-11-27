import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from assemble import header

def audit(official, model, selection, plan, binary):
    original_start, original = header(official)
    final_start, final = header(model / 'model-00001-of-00001.safetensors')
    recipe = json.loads(plan.read_text())
    modes = recipe['modes']
    cuts = json.loads(selection.read_text())['cutoffs']
    assert len(modes) == 40 and modes.keys() == cuts.keys()
    results = {}
    with TemporaryDirectory() as work:
        cutfile = Path(work) / 'cuts'
        for name, mode in sorted(modes.items()):
            cutfile.write_bytes(bytes(x if x >= 0 else 255 for x in cuts[name]))
            weight, scales = original[name], original[name+'_scale']
            args = [original_start + x['data_offsets'][0] for x in (weight, scales)]
            frames = []
            for suffix in ('.nvscale_direct_v1', '_scale.mean8_v1'):
                lo, hi = final[name+suffix]['data_offsets']
                frames.extend([final_start + lo, hi - lo])
            command = [binary, official, *map(str,args), model/'model-00001-of-00001.safetensors',
                       *map(str,frames), str(scales['shape'][-1]), cutfile, mode, recipe["palettes"][name], str(recipe["penalties"][name])]
            counts = list(map(int, subprocess.check_output(command, text=True).split()))
            assert len(counts) == 3 and sum(counts) == 128*32
            results[name] = dict(zip(('original','trim','reduced'), counts, strict=True))
    return results

if __name__ == '__main__':
    print(json.dumps(audit(*map(Path, sys.argv[1:])), indent=2))
