from pathlib import Path
import argparse, hashlib, json, shutil, struct, subprocess, tempfile

shard = "model-00001-of-00001.safetensors"
parser = argparse.ArgumentParser(
    description="Build two research-only Inkling MXFP8 precision variants from a canonical-zero package and official tensor codes."
)
parser.add_argument("official", type=Path)
parser.add_argument("parent", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument(
    "--encoder", type=Path, default=Path("target/release/examples/inkling_mxfp8_round")
)
parser.add_argument(
    "--cutoffs",
    type=Path,
    help="Use eight per-plane magnitude cutoffs from choose_cutoffs.py and write the budget variant.",
)
args = parser.parse_args()
thresholds = (
    json.loads(args.cutoffs.read_text())["thresholds"] if args.cutoffs else None
)
if thresholds is not None:
    keys = {
        f"model.llm.layers.{layer}.mlp.experts.{kind}"
        for layer in range(20, 24)
        for kind in ("w13_weight", "w2_weight")
    }
    if set(thresholds) != keys or any(
        type(v) != int or not 0 <= v <= 125 for v in thresholds.values()
    ):
        raise ValueError(
            "expected a finite magnitude cutoff for each of the eight planes"
        )
args.output_dir.mkdir(parents=True, exist_ok=True)
for arm in ("budget",) if thresholds else ("last", "all"):
    if (args.output_dir / arm).exists():
        raise FileExistsError(args.output_dir / arm)


def header(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return json.loads(f.read(n)), 8 + n


def copy(f, offset, n, out=None):
    f.seek(offset)
    h = hashlib.sha256()
    while n:
        part = f.read(min(n, 8 << 20))
        assert part
        if out is not None:
            out.write(part)
        h.update(part)
        n -= len(part)
    return h.hexdigest()


def number(m):
    return (m & 7) * 2**-9 if m < 8 else (8 + (m & 7)) * 2 ** ((m >> 3) - 10)


def rounded(m):
    options = range(0, 127, 2)
    return min(options, key=lambda k: (abs(number(k) - number(m)), (k // 2) % 2))


for m in range(127):
    expected = m if m % 2 == 0 else m + 1 if m & 2 else m - 1
    assert rounded(m) == expected
parent = args.parent
original = args.official / shard
with (parent / shard).open("rb") as f:
    parent_digest = hashlib.file_digest(f, "sha256").hexdigest()
if parent_digest != "9b242a81188f0f552625639dec98879a0090a7f9a003141631d3a836aed3fcb3":
    raise ValueError("expected the canonical-zero Inkling parent shard")
with original.open("rb") as f:
    raw, roff = header(f)
with (parent / shard).open("rb") as f:
    old, off = header(f)
frames = {}
with tempfile.TemporaryDirectory(prefix="round-frame-", dir=args.output_dir) as scratch:
    for layer in [23, 22, 21, 20]:
        for kind in ["w13_weight", "w2_weight"]:
            name = f"model.llm.layers.{layer}.mlp.experts.{kind}"
            key = name + ".rans_v1"
            a, b = raw[name]["data_offsets"]
            assert raw[name]["dtype"] == "F8_E4M3" and old[key]["dtype"] == "U8"
            cutoff = thresholds[name] if thresholds else 125
            dest = Path(scratch) / key
            lines = subprocess.check_output(
                [
                    str(args.encoder.resolve()),
                    str(original),
                    str(roff + a),
                    str(b - a),
                    str(dest),
                    str(cutoff),
                ],
                text=True,
            ).splitlines()
            hist = {int(v): int(n) for v, n in (line.split() for line in lines[1:])}
            assert (
                sum(hist.values()) == b - a
                and len(hist) == 256
                and hist[127] == hist[255] == 0
            )
            err = sum(
                n
                * (
                    number(i & 127)
                    - number(rounded(i & 127) if (i & 127) <= cutoff else i & 127)
                )
                ** 2
                for i, n in hist.items()
            )
            power = sum(n * number(i & 127) ** 2 for i, n in hist.items())
            frames[key] = dict(
                bytes=int(lines[0]),
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                changed_codes=sum(
                    n for v, n in hist.items() if v % 2 and (v & 127) <= cutoff
                ),
                codes=b - a,
                normalized_code_mse=err / power,
            )
            if thresholds:
                frames[key]["threshold"] = cutoff
        if layer not in ((20,) if thresholds else (20, 23)):
            continue
        arm = "budget" if thresholds else ("last" if layer == 23 else "all")
        target = Path(tempfile.mkdtemp(dir=args.output_dir, prefix=arm + ".partial."))
        meta = {}
        o = 0
        for k, v in old.items():
            if k == "__metadata__":
                meta[k] = v
                continue
            a, b = v["data_offsets"]
            length = frames[k]["bytes"] if k in frames else b - a
            z = v.copy()
            z["data_offsets"] = [o, o + length]
            if k in frames:
                z["shape"] = [length]
            meta[k] = z
            o += length
        data = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode()
        data += b" " * (-len(data) % 8)
        hashes = {}
        with (parent / shard).open("rb") as base, (target / shard).open("xb") as out:
            out.write(struct.pack("<Q", len(data)) + data)
            for k, v in meta.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                assert out.tell() == 8 + len(data) + a
                if k in frames:
                    with (Path(scratch) / k).open("rb") as f:
                        hashes[k] = copy(f, 0, b - a, out)
                    assert hashes[k] == frames[k]["sha256"]
                else:
                    hashes[k] = copy(base, off + old[k]["data_offsets"][0], b - a, out)
        with (target / shard).open("rb") as f:
            md, st = header(f)
            assert md == meta
            for k, h in hashes.items():
                a, b = meta[k]["data_offsets"]
                assert copy(f, st + a, b - a) == h
            f.seek(0)
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        shutil.copytree(parent / "tiktoken", target / "tiktoken")
        package = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        report = dict(
            raw_source="official Inkling-2-Mini",
            parent_sha256="9b242a81188f0f552625639dec98879a0090a7f9a003141631d3a836aed3fcb3",
            unchanged_tensors=len(hashes) - len(frames),
            shard_sha256=digest,
            package_bytes=package,
            selected=frames.copy(),
        )
        (args.output_dir / (arm + ".json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        target.rename(args.output_dir / arm)
        print(arm, package, digest, flush=True)
