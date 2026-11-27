# Choose the full-size BF16 rank split by physical bytes

The original entropy estimate chose five low bits. The encoder now invokes the existing production rANS packer for every feasible upper width and adds the exact table, header and lower-stream lengths. Only the winning low-bit stream is written. The loader already supports every measured width. The seven-bit result is **13,437,429,480 bytes** including the tokenizer. It saves **1,088,264 bytes** over the five-bit variant and **11,544,534 bytes** over the prefix parent. The remaining 998 tensor payloads and tokenizer were checked during assembly. [Encoder output](encoding.json) · [Actual model and tokenizer hashes](package.json).

| Low bits | Actual embedding frame bytes | Mean CPU unpack time, ms |
| ---: | ---: | ---: |
| 5 | 608,134,954 | 98.31 |
| 6 | 607,225,986 | 98.50 |
| 7 | 607,046,690 | 100.00 |
| 8 | 607,505,258 | 92.32 |

All four frames were physically written and decoded by the same unmodified PR reader. Every decoded BF16 word reproduces the publisher SHA-256 for 926,318,592 bytes. Forty fresh release processes rotated four arms and their reverse over five cycles on the Ryzen 9 9900X. The timers cover CPU decoding and exclude file I/O. The byte-aligned eight-bit package is **13,437,888,048 bytes**. It uses 458,568 more bytes than seven and has the shortest measured CPU decode. [Eight-bit distribution and tensor readback](byte-aligned-package.json). Against five bits, its paired time ratio is 0.9390 (95% whole-cycle interval 0.9180–0.9560). The seven-bit ratio is 1.0172 (0.9893–1.0514). [All widths, decoded hashes and per-process timings](widths.json).

To reproduce, use the [encoder, pack example and assembly instructions](../README.md#reproduce). Its default selects the smallest file. Use `--low-bits 8` for the byte-aligned alternative or another explicit width to reproduce a table entry. For CPU times build [decode.rs](../decode.rs) as a release leansonic-core example and run `rank_decode rank FRAME /dev/null` as a new process per leg. For cycles 1–4 rotate the initial order 5,6,7,8 by one position per cycle and append its reverse. Repeat the first order for the fifth cycle. Compare the complete decoded output and source hash separately.

A prototype specializing just the five-bit unpack loop in Rust took 113.32 ms versus 114.65 ms with the ordinary reader in a separate experiment. The paired 95% ratio interval, 0.960–1.018, crosses one. The default loader retains its existing generic loop. [Separate process timings](fixed-five.json).

The [small encoder check](check_widths.py) exhaustively compared automatic selection with every explicitly requested feasible width for two generated inputs. The first includes positive and negative zero, infinities and signed NaNs, with all eight widths tested. The second covers every possible BF16 bit pattern and exercises the 65,536-entry table at width eight. Every frame was reconstructed and compared byte for byte. Run in an empty directory with Python, NumPy, the rank packer and the [raw rANS readback example](../column-groups/unpack.rs):

```sh
python3 /path/to/check_widths.py /path/to/encode.py /path/to/rank_pack /path/to/context_unpack
```

[Per-width file sizes and reconstructed hashes](encoder-checks.json).

The normal CLI at 9bc4d48 loaded the seven-bit package on RTX PRO 4500. All 1,530 per-position fingerprints and score rows, plus the complete scorer JSON, match the previously tested five-bit package. Mean/worst KL 0.0085970/0.6397660, top-1 1,469/1,530 and PPL 6.564087. Use the six-chunk [GPU replay procedure](../README.md#gpu-replay) with the seven-bit directory. [GPU, model, binary and row hashes](gpu-4500.json).
