# Construct half-scaled FP32 values from six-bit NVFP4 scale fields

In the dominant packed gate/up decode kernel, this trial reads the two 6-bit fields used by each lane and constructs their exact half-scaled FP32 values from the exponent and mantissa. The existing arithmetic uses the original positive E4M3 byte (64 plus the code) and multiplies its value by 0.5. All 4,096 possible pairs at eight bit alignments, including word crossings, matched the existing device arithmetic bit for bit. PRO 4500 Compute Sanitizer reported zero errors for 32,768 comparisons. The packed kernel uses 62 registers versus 61 in the prior, with zero local and stack bytes in both.

| GPU | Phase | Prior ms | Candidate ms | Paired difference ms (95% interval) |
|---|---|---:|---:|---:|
| 5070Ti | prefill_ms | 121.9362 | 121.9237 | -0.0126 [-0.0330, +0.0036] |
| 5070Ti | decode32_ms | 192.3940 | 192.8304 | +0.4364 [+0.3674, +0.4907] |
| 5060Ti | prefill_ms | 257.4611 | 257.4756 | +0.0145 [-0.0214, +0.0445] |
| 5060Ti | decode32_ms | 339.5956 | 340.5305 | +0.9348 [+0.9000, +0.9772] |

Both boards reproduced all eight free-generation runs, including hashes of the final 200,058 FP32 logits. Both measured a decode regression with a positive paired interval. Keep the existing kernel. The register difference is an observation; these experiments do not establish a causal stall mechanism.

## Reproduce

Use the [two-scale trial replay procedure](https://github.com/tjm8874/LeanSonic/blob/247b160/docs/benchmarks/inkling-device-codebooks/decode-scale-pairs/README.md), with candidate cubin built at 69f6407 and prior at a567839d. Use the same integrated replay binary and compact distribution, FP16 KV, PDL, one sequence, 8,192 context, 1,024 prompt IDs and 32 forced IDs. Both arms used ten fresh processes, each with one excluded warm-up and four warm requests. The interval resamples five paired cycles.

Build the standalone device test on the candidate commit with `nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 docs/benchmarks/inkling-device-codebooks/decode-scale-exponents/check.cu -o check`. Run `compute-sanitizer --tool memcheck --error-exitcode 4 ./check`. [Source hashes, memcheck, parity traces and individual phase times](receipt.json).
