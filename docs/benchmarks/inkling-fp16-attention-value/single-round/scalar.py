"""Check online weighted sums against independently rounded exact rationals."""
import collections
import ctypes
import json
import math
import random
import struct
from fractions import Fraction

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]
def bits(x):
    return struct.unpack('I', struct.pack('f', x))[0]
def value(b):
    return struct.unpack('f', struct.pack('I', b))[0]
fma = ctypes.CDLL(None).fmaf
fma.argtypes = [ctypes.c_float] * 3
fma.restype = ctypes.c_float
rng = random.Random(229)
stats = {}
for case in ('old maximum','new maximum','tie'):
    counts = collections.Counter()
    for i in range(16384):
        prior=f32(rng.uniform(-8,8)*2**rng.randrange(-10,8))
        val=struct.unpack('e', struct.pack('e',rng.uniform(-4,4)*2**rng.randrange(-10,8)))[0]
        weight=f32(math.exp(-rng.uniform(0,24)))
        alpha=weight if case=='new maximum' else 1.
        p=weight if case=='old maximum' else 1.
        truth=Fraction(prior)*Fraction(alpha)+Fraction(p)*Fraction(val)
        mid=bits(float(truth))
        near=[k for k in (mid-1,mid,mid+1) if 0<=k<2**32 and math.isfinite(value(k))]
        target=value(min(near,key=lambda k:(abs(Fraction(value(k))-truth),k%2)))
        outputs={'current':fma(alpha,prior,f32(p*val)), 'pr229':fma(p,val,f32(prior*alpha)),
                 'single_round':fma(p,val,prior) if alpha==1. else fma(alpha,prior,val)}
        for mode,result in outputs.items():counts[mode]+=int(result!=target)
    assert counts['single_round']==0
    stats[case]=dict(rows=16384,rounding_errors=counts)
print(json.dumps(stats,indent=2))
