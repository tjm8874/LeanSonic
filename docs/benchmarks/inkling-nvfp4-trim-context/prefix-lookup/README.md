# CPU decode of the exact trim-context frames

A 4 KiB table maps the preceding context and next NVFP4 byte to the next context. It changes only the decoder. All 4,096 entries are tested against the encoder transition. Two real frames from the [13,445,948,726-byte package](../README.md) were decoded on an AMD Ryzen 9 9900X with four logical CPUs (Linux affinity 4–7). Each run read the frame from the real shard before timing, decoded once for a reference, and measured three more complete decodes including output allocation. Each full output equaled the reference. The per-frame hash also agreed across builds.

| NVFP4 plane | Decoded B | Control ms | Lookup ms | Paired difference ms (95% interval) |
|---|---:|---:|---:|---:|
| Layer 0 down | 169,869,312 | 198.275 | 163.749 | −34.526 [−44.153, −24.900] |
| Layer 11 gate/up | 339,738,624 | 381.385 | 289.401 | −91.984 [−120.112, −56.943] |

Five alternating ABBA/BAAB cycles contributed ten fresh processes per arm per plane. Intervals resample the five paired cycle means. The source builds are 877a4f5 and e31d8ac. To repeat, build [decode.rs](decode.rs) as a leansonic-core release example in each checkout. Pass the safetensors shard, absolute start byte of the tagged plane, encoded length and decoded length. The start byte is eight plus the JSON header length plus the tensor's first data offset. Limit both programs to the same four CPUs. [Individual timings and output fingerprints](verified.json).

## Complete model load on RTX PRO 4500

With the 13,445,948,726-byte trim package held fixed, the observed model loader time fell from **7.790 s to 6.863 s** (−0.927 s, paired 95% interval −1.170 to −0.657). The loader uses the host CPU to decode expert frames before uploading weights. These times cover fresh processes with normal OS file caching. They do not measure a cold filesystem.

Both arms used the same GPU cubin and the same fixed [parity driver](../parity.rs). The source builds are 8c77955 (integrated storage loader) and 173fe9c (plus the table). Use the [runtime environment and original README ID file](../runtime.md). Start five alternating ABBA/BAAB cycles with two processes per arm per cycle. All 160 requests agreed on 64 greedy IDs, next ID and final 200,058-logit hash at inputs of 32 and 1,024 tokens. At 1,024 input tokens, warm prefill was 122.529 vs 122.537 ms (difference +0.008, interval −0.010 to +0.025); 64-token decode was 423.173 vs 423.066 ms (−0.106, interval −0.290 to +0.031). [Each loader observation and generated trace](loader.json).
