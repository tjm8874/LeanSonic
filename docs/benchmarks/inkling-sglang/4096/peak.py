"""Sample only the target job and its descendant compute processes via NVML."""
import ctypes as ct
import json
import os
import sys
import time
from pathlib import Path

pid=int(Path(sys.argv[1]).read_text())
dest=Path(sys.argv[2])
def proc_info(p):
    try:
        rest=Path(f"/proc/{p}/stat").read_text().rsplit(")",1)[1].split()
        return int(rest[1]),int(rest[19])
    except (FileNotFoundError,ProcessLookupError):
        return 0,0
root_start=proc_info(pid)[1]
assert root_start
def belongs(p):
    while p>1 and p!=pid:
        p=proc_info(p)[0]
    return p==pid
lib=ct.CDLL("libnvidia-ml.so.1")
assert lib.nvmlInit_v2()==0
dev=ct.c_void_p()
import subprocess
uid=os.environ['CUDA_VISIBLE_DEVICES']
assert lib.nvmlDeviceGetHandleByUUID(uid.encode(),ct.byref(dev))==0
class Compute(ct.Structure):
    _fields_=[("pid",ct.c_uint),("bytes",ct.c_ulonglong),("gi",ct.c_uint),("ci",ct.c_uint)]
buf=(Compute*64)()
maximum=0
reads=0
observed=0
while proc_info(pid)[1]==root_start:
    count=ct.c_uint(64)
    assert lib.nvmlDeviceGetComputeRunningProcesses_v2(dev,ct.byref(count),buf)==0
    total=sum(s.bytes for s in buf[:count.value] if belongs(s.pid))
    assert total<2**60
    maximum=max(maximum,total)
    observed+=int(total>0)
    reads+=1
    time.sleep(.004 if total else .05)
dest.write_text(json.dumps(dict(peak_bytes=maximum,polls=reads,positive_polls=observed))+"\n")
