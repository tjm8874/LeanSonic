"""Create a standalone model distribution from the compact package and fitted tables."""

import hashlib, json, os, sys, zipfile
from pathlib import Path

base, tables, dest = map(Path, sys.argv[1:])
fit = json.loads((tables / "fit.json").read_text())
if len(fit) != 8:
    raise ValueError("eight planes required")
dest.mkdir()
for src in sorted(base.rglob("*")):
    out = dest / src.relative_to(base)
    if src.is_dir():
        out.mkdir()
    elif src.is_file():
        os.link(src, out)
    else:
        raise ValueError(src)
patch = dest / "mxfp8-group-tables.zip"
with zipfile.ZipFile(
    patch, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
) as z:
    for name in sorted(["fit.json"] + [k + ".lut" for k in fit]):
        data = (tables / name).read_bytes()
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
with zipfile.ZipFile(patch) as z:
    assert z.testzip() is None and all(
        z.read(k + ".lut") == (tables / (k + ".lut")).read_bytes() for k in fit
    )
files = {
    f.relative_to(dest).as_posix(): f.stat().st_size
    for f in dest.rglob("*")
    if f.is_file()
}
print(
    json.dumps(
        dict(
            package_bytes=sum(files.values()),
            files=files,
            sidecar_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
        )
    ),
    flush=True,
)
