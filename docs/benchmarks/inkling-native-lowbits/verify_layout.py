"""Compare the logical tensor shapes and byte-serialized tokenizer metadata of GGUF v3 files."""
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def inspect(path):
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        if m.read(4) != b'GGUF':
            raise ValueError(f'Invalid magic in {path.name}')

        def number(fmt):
            return struct.unpack(fmt, m.read(struct.calcsize(fmt)))[0]

        def string():
            return m.read(number('<Q')).decode('utf-8')

        def skip(kind, count=1):
            if kind in WIDTH:
                m.seek(WIDTH[kind] * count, 1)
            elif kind == 8:
                for _ in range(count):
                    m.seek(number('<Q'), 1)
            elif kind == 9:
                for _ in range(count):
                    subtype, n = struct.unpack('<IQ', m.read(12))
                    skip(subtype, n)
            else:
                raise ValueError(f'Unknown GGUF field type {kind}')

        version, tensors, fields = struct.unpack('<IQQ', m.read(20))
        assert version == 3
        meta = {}
        architecture = None
        for _ in range(fields):
            key = string()
            start = m.tell()
            kind = number('<I')
            if key == 'general.architecture':
                assert kind == 8
                architecture = string()
            else:
                skip(kind)
            assert key not in meta
            meta[key] = hashlib.sha256(m[start:m.tell()]).hexdigest()
        assert architecture and 'tokenizer.ggml.tokens' in meta
        shapes = {}
        for _ in range(tensors):
            name = string()
            nd = number('<I')
            assert 1 <= nd <= 4 and name not in shapes
            shapes[name] = list(struct.unpack('<' + 'Q'*nd, m.read(8*nd)))
            m.seek(12, 1)  # Storage type and byte offset differ with quantization.
        dimensions = json.dumps(shapes, sort_keys=True, separators=(',', ':')).encode()
        return dict(architecture=architecture, tensor_count=tensors,
                    tensor_shapes_sha256=hashlib.sha256(dimensions).hexdigest(),
                    tokenizer={k:v for k,v in meta.items() if k.startswith('tokenizer.')},
                    model_parameters={k:v for k,v in meta.items() if k.startswith(architecture+'.')})

files = [Path(p) for p in sys.argv[1:]]
if len(files) < 2:
    raise SystemExit('usage: verify_layout.py REFERENCE.gguf CANDIDATE.gguf [...]')
baseline = inspect(files[0])
for candidate in files[1:]:
    current = inspect(candidate)
    for key, ref in baseline.items():
        if current[key] != ref:
            raise ValueError(f'{candidate.name} differs from {files[0].name} in {key}')
print(json.dumps(dict(files=[f.name for f in files], **baseline), indent=2))
