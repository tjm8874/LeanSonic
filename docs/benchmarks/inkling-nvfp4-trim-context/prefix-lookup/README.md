# CPU decode of the exact trim-context frames

A 4 KiB table maps the preceding context and next NVFP4 byte to the next context. It changes only the decoder. All 4,096 entries are tested against the encoder transition. Two real frames from the [13,445,948,726-byte package](../README.md) were decoded on an AMD Ryzen 9 9900X with four logical CPUs (Linux affinity 4–7). Each run read the frame from the real shard before timing, decoded once for a reference, and measured three more complete decodes including output allocation. Each full output equaled the reference. The per-frame hash also agreed across builds.

| NVFP4 plane | Decoded B | Control ms | Lookup ms | Paired difference ms (95% interval) |
|---|---:|---:|---:|---:|
| Layer 0 down | 169,869,312 | 198.275 | 163.749 | −34.526 [−44.153, −24.900] |
| Layer 11 gate/up | 339,738,624 | 381.385 | 289.401 | −91.984 [−120.112, −56.943] |

Five alternating ABBA/BAAB cycles contributed ten fresh processes per arm per plane. Intervals resample the five paired cycle means. The source builds are 877a4f5 and e31d8ac. To repeat, build [decode.rs](decode.rs) as a leansonic-core release example in each checkout. Pass the safetensors shard, absolute start byte of the tagged plane, encoded length and decoded length. The start byte is eight plus the JSON header length plus the tensor's first data offset. Limit both programs to the same four CPUs. [Individual timings and output fingerprints](verified.json).
