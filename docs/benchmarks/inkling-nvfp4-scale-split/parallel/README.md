# Parallel placement of scale-selected Inkling blocks

The two expert streams and cutoffs use the same format as the [scale-selected storage experiment](../README.md). This trial computes per-expert start offsets in each stream, then writes distinct ranges of the reconstructed output concurrently. The two real NVFP4 planes come from the 13,479,712,078-byte package. Each process decoded the existing mean scale frame before timing and held it fixed.

On a Ryzen 9 9900X pinned to four logical CPUs, layer 0 down decoded in 309.026 ms before and 216.911 ms after the change (paired difference −92.116 ms, 95% interval −124.251 to −67.123). Layer 11 gate/up took 502.661 vs 425.748 ms (−76.914, interval −137.926 to +23.535). The longer plane's interval spans zero. Each arm had ten processes and three timed full decodes per process. All reconstructed bytes equaled the warm reference and the output fingerprint matched across arms for each plane.

Use [decode.rs](decode.rs) as a release example. Pass shard path, weight offset, weight length, decoded weight bytes, scale offset, encoded scale length and original scale row width. Use five alternating ABBA/BAAB cycles. [All observations](verified.json). 

## Full-engine load

Using the same 13,479,712,078-byte package, five alternating ABBA/BAAB cycles on RTX PRO 4500 reduced mean model load from 15.680 to 12.526 s (paired change −3.154 s, cycle-bootstrap 95% interval −3.930 to −2.218). Maximum process RSS across ten starts per arm was 6,650,984 KiB serial and 6,650,656 KiB parallel. For a 1,024-token prefix, warm prefill averaged 119.706 vs 119.799 ms and 64-token decode averaged 421.707 vs 421.866 ms. All 160 requests matched their prefix-specific greedy IDs, next ID and final FP32 logit hash.

Compare integrated commits 9ecca08 and eaa1813 with the same bc526e3 cubin, FP16 KV, scratch-free U4, spec 0, one sequence, context 2,048. Run the [parity driver](https://github.com/tjm8874/LeanSonic/tree/cc62564/docs/benchmarks/inkling-nvfp4-trim-context) four times per process at 32 and 1,024 tokens, and record host RSS using GNU time %M. [Individual observations and binary hashes](engine.json).
