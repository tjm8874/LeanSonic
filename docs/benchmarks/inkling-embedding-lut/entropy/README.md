# Entropy coding for the 217-expert Inkling input embedding

The 217-expert model and the same model with entropy-coded indices occupy **11,662,534,370 and 11,614,890,622 bytes**. The 47,643,748-byte saving changes only the representation of the input embedding. Reconstructing all 463,159,296 BF16 values produced the same 926,318,592 bytes and SHA-256 775deb5e5a2784aa028f0af28dffe05c38538497dbc6b9f494d78e91bfc79099. The other 998 tensor payloads and the tokenizer matched byte for byte.

The 512-byte BF16 lookup table is followed by four little-endian u32s giving the literal count and three frame lengths. The three rANS streams encode the indices, low literal bytes and high literal bytes, using the existing tiled byte codec. When no literals occur both literal streams have zero length. The loader validates the dimensions, tables, lengths and finite literals, and writes BF16 into its final pinned buffer. Index decoding is streamed in blocks. GPU weight representation and the inference kernels are identical between the two packages.

| Full input embedding | Bytes | Warm mean decode on CPU |
|---|---:|---:|
| Indices and uncompressed literals | 502,097,692 | 231.0 ms |
| Three rANS streams | 454,453,936 | 331.2 ms |

Both complete decodes ran in five mirrored four-leg cycles on the same host. The warm means exclude the first entire cycle, leaving eight observations per arm. The absolute figures depend on host load. This code trades roughly 100 ms of CPU unpacking at model load for the 47.64 MB reduction. No steady-state speed improvement is claimed. A six-chunk GPU comparison of the two physical packages is queued with matching runtime and module.

To reproduce, use d15b1d69f38eac343fb8cda80695552cb1266694. Build probe.rs as a leansonic-core release example and run it with the original bounded.lut, a new coded.lut path and a decoded.bf16 path. Compare the digest with proof.sha. Then run the parent directory's assemble.py with the original [217-expert model](../reallocated/), the coded file, an unused output directory and --entropy. The package digest is in [receipt.json](receipt.json); CPU decode times are in [timing.csv](timing.csv). The [fixed-capacity refit](refit/) buys additional embedding precision with the recovered bytes.
