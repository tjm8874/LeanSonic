"""Build both measured class-frame readers in a temporary source tree."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source_repo = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
candidate = "38e8d57"
initial = "ca0fff1"
module = "crates/leansonic-core/src/inkling_mxfp8.rs"
signature = "pub(crate) fn unpack_class_scale_banks"
end_marker = "fn pack_nvfp4_tables"

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    archive = root / "source.tar"
    subprocess.run(
        ["git", "-C", source_repo, "archive", "-o", archive, candidate], check=True
    )
    subprocess.run(["tar", "-xf", archive, "-C", root], check=True)
    old = subprocess.check_output(
        ["git", "-C", source_repo, "show", f"{initial}:{module}"], text=True
    )
    start = old.index(signature)
    end = old.index(end_marker, start)
    prior_reader = old[start:end].replace(signature, "pub fn benchmark_prior_class_frame")
    path = root / module
    current = path.read_text()
    assert current.count(signature) == 1
    path.write_text(current.replace(signature, "pub fn unpack_class_scale_banks") + prior_reader)
    example = "class_scatter_bench"
    shutil.copyfile(
        Path(__file__).with_name("bench.rs"),
        root / "crates/leansonic-core/examples" / f"{example}.rs",
    )
    target = root / "target"
    subprocess.run(
        ["cargo", "build", "--release", "--target-dir", target, "-p", "leansonic-core", "--example", example],
        cwd=root, check=True,
    )
    shutil.copy2(target / "release/examples" / example, output)
