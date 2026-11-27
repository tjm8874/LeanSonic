"""Rank complete expert pairs by summed training gain per estimated byte."""
from collections import defaultdict
import json
from pathlib import Path
import sys
source, destination = map(Path, sys.argv[1:])
record = json.loads(source.read_text())
groups = defaultdict(list)
for entry in record['rank']:
    name = entry['plane'].split('.')
    groups[int(name[3]), entry['expert']].append(entry)
items = []
for (layer, expert), pair in groups.items():
    projections = {e['plane'].rsplit('.', 1)[1] for e in pair}
    expected = {'w13_weight'} if layer in (16, 17) else {'w13_weight', 'w2_weight'}
    assert projections == expected and len(pair) == len(expected)
    score = sum(e['gain'] for e in pair) / sum(e['estimated_bytes'] for e in pair)
    items.append((-score, layer, expert, sorted(pair, key=lambda x: x['plane'])))
rank = []
ends = [0]
for _, _, _, pair in sorted(items):
    rank.extend(pair)
    ends.append(len(rank))
assert len(rank) == 4864 and len(ends) == 2561
record.update(rank=rank, legal_ends=ends, approach='training gain per byte of complete w13/w2 pairs; singleton w13 on layers 16 and 17 with previously restored w2')
destination.write_text(json.dumps(record, separators=(',', ':'))+'\n')
