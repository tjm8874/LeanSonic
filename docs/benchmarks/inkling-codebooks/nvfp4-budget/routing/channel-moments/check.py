from pathlib import Path
import hashlib
import json

import numpy as np

for name, digest in {
    'windows.ids':'825d5a5283d3ebfe36901051db221a52c6bfbcd26cf097a01bfa6e28f6fb24fa',
    'routes.bin':'900f4ceffd4e0cb847ce96bd57ea9a401d4f996531af71a09f76ed0b9dc3a3f5',
    'predictions':'2bc8f2f0a94e423d96bc33ea9d6eba043b0f63bcf9db2e2da4c4eb8cbd801579',
}.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
a=np.fromfile('moments.f64','<f8').reshape(20,128,2304)
counts=np.array(json.loads(Path('routing.json').read_text())['hits'])
assert np.all(np.isfinite(a)) and np.all(a>=0)
assert counts.sum()==5767168 and np.array_equal(a.sum(axis=-1)>0,counts>0)
present=a[counts>0]
print(json.dumps(dict(shape=list(a.shape),bytes=a.nbytes,sha256=hashlib.sha256(Path('moments.f64').read_bytes()).hexdigest(),routed_selections=int(counts.sum()),experts_seen=int((counts>0).sum()),predictions_identical=True,routing_identical=True,train_only=True,
    median_channel_p90_over_p10=float(np.median(np.quantile(present,.9,axis=1)/np.quantile(present,.1,axis=1)))),indent=2))
