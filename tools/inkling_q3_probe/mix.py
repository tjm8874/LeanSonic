"""Use raw tensor bytes from two pinned, layout-compatible original-release GGUF files."""

from pathlib import Path
import hashlib, json, os, struct, sys
import gguf

iq4, q3, out = map(Path, sys.argv[1:4])
layers = tuple(map(int, sys.argv[4:]))
assert layers and len(layers) == len(set(layers))
a, b = gguf.GGUFReader(str(iq4)), gguf.GGUFReader(str(q3))
aa = {x.name: x for x in a.tensors}
bb = {x.name: x for x in b.tensors}
assert aa.keys() == bb.keys() and a.alignment == b.alignment and a.alignment == 32
names = set()
for layer in layers:
    prefix = f"blk.{layer}.ffn_gate_up_exps."
    group = {n for n in aa if n.startswith(prefix)}
    assert (
        prefix + "weight" in group
        and aa[prefix + "weight"].tensor_type.name == "NVFP4"
        and bb[prefix + "weight"].tensor_type.name == "IQ2_S"
    )
    names.update(group)
chunksize = 4 << 20


def copy(src, dst, n, digest):
    while n:
        data = src.read(min(chunksize, n))
        assert data
        digest.update(data)
        if dst is not None:
            dst.write(data)
        n -= len(data)


chosen = {n: (bb[n] if n in names else aa[n]) for n in aa}
tmp = out.with_suffix(".part")
assert not out.exists()
with iq4.open("rb") as fa, q3.open("rb") as fb, tmp.open("xb") as fo:
    header = bytearray(fa.read(a.data_offset))
    data_order = sorted(aa, key=lambda n: aa[n].data_offset)
    offset = 0
    expected = {}
    offsets = {}
    for name in data_order:
        old = aa[name]
        new = chosen[name]
        assert list(old.shape) == list(new.shape)
        pos = old.field.offset + sum(part.nbytes for part in old.field.parts[:-2])
        assert struct.unpack_from("<IQ", header, pos) == (
            int(old.tensor_type),
            old.data_offset - a.data_offset,
        )
        struct.pack_into("<IQ", header, pos, int(new.tensor_type), offset)
        offsets[name] = offset
        offset = (offset + new.n_bytes + a.alignment - 1) // a.alignment * a.alignment
    fo.write(header)
    for name in data_order:
        src = fb if name in names else fa
        t = chosen[name]
        src.seek(t.data_offset)
        fo.write(bytes(a.data_offset + offsets[name] - fo.tell()))
        digest = hashlib.sha256()
        copy(src, fo, t.n_bytes, digest)
        expected[name] = digest.hexdigest()
    end = aa[data_order[-1]].data_offset + aa[data_order[-1]].n_bytes
    padding = iq4.stat().st_size - end
    assert 0 <= padding < a.alignment
    fo.write(bytes(padding))
r = gguf.GGUFReader(str(tmp))
assert len(r.tensors) == len(chosen)
with tmp.open("rb") as check:
    for t in r.tensors:
        orig = chosen[t.name]
        assert (
            t.tensor_type == orig.tensor_type
            and t.n_bytes == orig.n_bytes
            and list(t.shape) == list(orig.shape)
        )
        check.seek(t.data_offset)
        digest = hashlib.sha256()
        copy(check, None, t.n_bytes, digest)
        assert digest.hexdigest() == expected[t.name], t.name
diff = sum(aa[n].n_bytes - bb[n].n_bytes for n in names)
assert iq4.stat().st_size - tmp.stat().st_size == diff
os.link(tmp, out)
tmp.unlink()
print(
    json.dumps(
        dict(
            tensors=len(chosen),
            layers=layers,
            bytes=out.stat().st_size,
            saved=diff,
            selected={
                n: {
                    "source": aa[n].tensor_type.name,
                    "target": bb[n].tensor_type.name,
                    "sha256": expected[n],
                }
                for n in sorted(names)
            },
        )
    )
)
