import json
import sqlite3

db = sqlite3.connect("prefill.sqlite")
raw = list(
    db.execute(
        "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"
    )
)
assert raw and all(end >= start for start, end, _ in raw)
totals = {}
busy = 0
last = raw[0][0]
for start, end, name in raw:
    row = totals.setdefault(name, [0, 0])
    row[0] += 1
    row[1] += end - start
    busy += max(0, end - max(last, start))
    last = max(last, end)
elapsed = max(end for _, end, _ in raw) - raw[0][0]
print(
    json.dumps(
        {
            "launches": len(raw),
            "span_ms": elapsed / 1e6,
            "kernel_busy_ms": busy / 1e6,
            "uncovered_ms": (elapsed - busy) / 1e6,
            "kernels": [
                {"name": k, "count": n, "total_ms": t / 1e6}
                for k, (n, t) in sorted(totals.items(), key=lambda v: -v[1][1])
            ],
        },
        indent=2,
    )
)
