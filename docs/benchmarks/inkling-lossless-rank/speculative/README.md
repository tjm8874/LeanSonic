# Greedy generation with the ranked four-bank target

All three arms use the same 13,438,517,744-byte five-bit target and tokenizer, built at 9bc4d48. DSpark and DFlash 2 each add their own drafter weights. One sequence, 2,048-token context, FP16 KV, 1,024 prompt tokens, 64 greedy output tokens, no early EOS. Each run returns a 65th token to seed the next step. The solo and speculative paths produced identical output IDs and identical next tokens on all four fixtures in every completed leg.

**1 of three mirrored cycles reported.** Two processes per arm per cycle. Each fresh process evaluates all four prompts four times with an engine reset. The table excludes the first repetition per fixture. On RTX PRO 4500 all 24 target expert layers stayed resident.

| Arm | Model and drafter bytes | Peak GPU bytes | README prefill / decode | Test 0 | Test 35 | Valid 84 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solo | 13,438,517,744 | 15,590,227,968 | 8,555.6 / 123.47 | 8,834.1 / 123.45 | 8,796.8 / 123.45 | 8,772.3 / 123.42 |
| DSpark K=7 | 14,074,319,160 | 16,160,653,312 | 8,139.1 / 130.80 | 8,229.8 / 220.30 | 8,432.6 / 132.16 | 8,422.8 / 121.61 |
| DFlash 2 K=7 | 14,428,493,360 | 16,651,386,880 | 7,821.6 / 191.33 | 8,032.6 / 234.60 | 8,039.2 / 169.69 | 8,095.1 / 130.97 |

All phase rates are tokens per second. Each cell uses the per-request token count divided by the mean wall-clock phase duration for that fixture and arm. Launch and weight loading are outside the phase timers. The NVML peak includes initialization. A 64-token free-generation rate and a 32-step teacher-forced rate on the public-engine replay are different workloads.

## Reproduce

Build [generate.rs](generate.rs) as a release leansonic-core example at 9bc4d48. Use the five-bit [rank distribution](../receipt.json), the same SM120a cubin as the [public-engine comparison](../vllm/README.md), and the four [full-logit fixture inputs](https://github.com/tjm8874/LeanSonic/tree/78def8f/docs/benchmarks/inkling-class-frames/three-engines/prefill-tiles). Each fixture line has 1,056 comma-separated IDs. The benchmark consumes the first 1,024 and runs free greedy generation. The readme fixture uses the common engine prompt. The other three start at test chunks 0 and 35 and validation chunk 84.

For solo run `LEANSONIC_DRAFTER=auto generate MODEL 0 fixtures.ids`. For DSpark add the official 635,801,416-byte mtp.safetensors to the target package and run `LEANSONIC_DRAFTER=dspark generate MODEL_WITH_MTP 7 fixtures.ids`. For DFlash 2 use the 989,975,616-byte checkpoint with `LEANSONIC_DRAFTER=dflash2 LEANSONIC_DFLASH2=/path/to/model.safetensors generate MODEL 7 fixtures.ids`. Enable LEANSONIC_PDL, LEANSONIC_PDL_ALL, LEANSONIC_MXFP8_DEVICE_U4 and LEANSONIC_MXFP8_DIRECT_ALL. Set LEANSONIC_KV=fp16, LEANSONIC_SEQS=1, LEANSONIC_CTX=2048, LEANSONIC_PREFILL_CHUNK=1024, and LEANSONIC_CUBIN_PATH to the confirmed cubin. Rotate the initial solo, DSpark, DFlash 2 order by one per cycle and append its reverse. The [receipt](replay.json.gz) includes executable and weight hashes, all reference output IDs, next-token checks, per-process peaks, per-fixture times and draft acceptance counts.
