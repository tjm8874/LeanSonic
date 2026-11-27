# Full inference with byte-permutation unpacking

Both arms used the same merged gate/up host, two-bit gate/up and down, actual 7,718,950,382-byte package, 4,096 context, one sequence, FP16 KV, no drafter. Only the sm_120a module differed. Five mirrored ABBA/BAAB cycles per GPU gave 30 warm requests per prefix and arm. All 240 requests, including warmups, returned identical 64-token predictions, successors and final FP32 logit fingerprints. Intervals resample whole cycles.

## 5070Ti

| Prefix | Prefill ms, prior / trial | Decode tok/s, prior / trial | Decode 64 change ms [95%] | Prefill + decode 64 change ms [95%] |
|---:|---:|---:|---:|---:|
| 32 | 28.205 / 27.938 | 203.45 / 203.32 | +0.211 [+0.159, +0.266] | -0.056 [-0.118, +0.007] |
| 1024 | 119.997 / 119.481 | 142.76 / 142.67 | +0.280 [+0.196, +0.364] | -0.236 [-0.332, -0.137] |
| 2048 | 210.381 / 209.816 | 135.93 / 135.84 | +0.311 [+0.221, +0.409] | -0.254 [-0.404, -0.100] |

Peak observed process VRAM was 11,813,257,216 / 12,064,915,456 bytes. [Per-request times, predictions, module hashes and peaks](5070Ti.json).

## 5060Ti

| Prefix | Prefill ms, prior / trial | Decode tok/s, prior / trial | Decode 64 change ms [95%] | Prefill + decode 64 change ms [95%] |
|---:|---:|---:|---:|---:|
| 32 | 49.506 / 48.925 | 116.49 / 116.45 | +0.185 [+0.050, +0.320] | -0.395 [-0.554, -0.236] |
| 1024 | 238.269 / 237.195 | 93.52 / 93.50 | +0.127 [-0.131, +0.415] | -0.946 [-1.278, -0.574] |
| 2048 | 428.545 / 427.363 | 90.60 / 90.59 | +0.139 [-0.226, +0.534] | -1.043 [-1.448, -0.605] |

Peak observed process VRAM was 11,966,349,312 / 11,865,686,016 bytes. [Per-request times, predictions, module hashes and peaks](5060Ti.json).

The two modules contain 234 named SASS functions. Only the four grouped ternary Tensor Core functions and the fallback expander changed instruction words. All four ternary DP4A decode variants have identical instruction words and scheduling-control bits. This leaves the cause of any small decode timing shift open. [SASS comparison](sass-diff.json).

Reproduce with the [merged timing driver](../../merged-timing/) using LEANSONIC_TERNARY_UP=1 and LEANSONIC_TERNARY_DOWN=1 in both arms. Change only LEANSONIC_CUBIN_PATH between the original module and [the lookup experiment](../).
