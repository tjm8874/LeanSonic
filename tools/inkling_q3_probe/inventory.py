import collections, json, re, sys
from pathlib import Path
import gguf


def layout(path):
    r = gguf.GGUFReader(path)
    named = {
        t.name: (t.tensor_type.name, int(t.n_bytes), list(map(int, t.shape)))
        for t in r.tensors
    }
    assert len(named) == len(r.tensors)
    size = Path(path).stat().st_size
    payload = sum(t[1] for t in named.values())
    assert payload <= size
    return named, size - payload, size


a, ma, sa = layout(sys.argv[1])
b, mb, sb = layout(sys.argv[2])
changes = []
groups = collections.defaultdict(lambda: [0, 0, 0])
for name in sorted(a.keys() | b.keys()):
    at, ab, ashape = a.get(name, ("absent", 0, []))
    bt, bb, bshape = b.get(name, ("absent", 0, []))
    assert not ashape or not bshape or ashape == bshape, (name, ashape, bshape)
    if ab == bb and at == bt:
        continue
    family = re.sub(r"^blk\.\d+\.", "blk.*.", name)
    group = groups[(family, at, bt)]
    group[0] += 1
    group[1] += ab
    group[2] += bb
    changes.append(
        dict(
            name=name,
            iq4_type=at,
            q3_type=bt,
            iq4_bytes=ab,
            q3_bytes=bb,
            diff_bytes=ab - bb,
        )
    )
grouped = [
    dict(
        family=f,
        iq4_type=x,
        q3_type=y,
        count=n,
        iq4_bytes=v,
        q3_bytes=w,
        diff_bytes=v - w,
    )
    for (f, x, y), (n, v, w) in groups.items()
]
grouped.sort(key=lambda row: -row["diff_bytes"])
assert sum(x["diff_bytes"] for x in grouped) + ma - mb == sa - sb
report = dict(
    iq4_bytes=sa,
    q3_bytes=sb,
    iq4_overhead=ma,
    q3_overhead=mb,
    iq4_tensor_count=len(a),
    q3_tensor_count=len(b),
    groups=grouped,
    changed_tensors=changes,
)
Path("q3-inventory.json").write_text(json.dumps(report, indent=2) + "\n")
print(sa, sb, ma, mb, len(a), len(b))
for x in grouped[:8]:
    print(x["family"], x["iq4_type"], x["q3_type"], x["count"], x["diff_bytes"])
