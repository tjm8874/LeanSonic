# Parallel placement of scale-selected Inkling blocks

The two expert streams and cutoffs use the same format as the [scale-selected storage experiment](../README.md). This trial computes per-expert start offsets in each stream, then writes distinct ranges of the reconstructed output concurrently. The two real NVFP4 planes come from the 13,479,712,078-byte package. Each process decoded the existing mean scale frame before timing and held it fixed.

On a Ryzen 9 9900X pinned to four logical CPUs, layer 0 down decoded in 309.026 ms before and 216.911 ms after the change (paired difference −92.116 ms, 95% interval −124.251 to −67.123). Layer 11 gate/up took 502.661 vs 425.748 ms (−76.914, interval −137.926 to +23.535). The longer plane's interval spans zero. Each arm had ten processes and three timed full decodes per process. All reconstructed bytes equaled the warm reference and the output fingerprint matched across arms for each plane.

Use [decode.rs](decode.rs) as a release example. Pass shard path, weight offset, weight length, decoded weight bytes, scale offset, encoded scale length and original scale row width. Use five alternating ABBA/BAAB cycles. [All observations](verified.json). A separate full-engine ABBA measures startup time and process memory.
