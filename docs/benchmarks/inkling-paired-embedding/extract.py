import hashlib
import json
import struct
import sys
from pathlib import Path

for src, dest, size, digest in (
    (sys.argv[1], 'official.bf16', 926318592, '86afe04b91893883cf823392f1c954bbf0881988e0b6c5f34ed7b2f938fe0a30'),
    (sys.argv[2], 'prefix.bin', 618590352, '9c9ac17b695cf5eeedfb18fba930c82d0fcd883c3f8619e251a57619db556f6a'),
):
    with Path(src).open('rb') as f:
        header_length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_length))
        entries = [(name, v['data_offsets']) for name, v in header.items() if name in ('model.llm.embed.weight', 'model.llm.embed.weight.prefix_v1') and 'data_offsets' in v and v['data_offsets'][1] - v['data_offsets'][0] == size]
        assert len(entries) == 1, entries
        name, (a,b) = entries[0]
        f.seek(8 + header_length + a)
        h = hashlib.sha256()
        with Path(dest).open('xb') as out:
            left = b - a
            while left:
                data = f.read(min(left, 8 << 20))
                assert data
                h.update(data)
                out.write(data)
                left -= len(data)
        print(name, h.hexdigest(), flush=True)
        assert h.hexdigest() == digest
