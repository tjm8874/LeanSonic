# Optional VRAM margin on 16 GB Blackwell

Set `LEANSONIC_VRAM_RESERVE_MIB` to a nonnegative whole number of MiB. The loader subtracts this margin from the free memory it uses to choose expert placement, prefill workspace and host-transfer staging. The default is zero. Weights/KV already allocated, mandatory workspace, CUDA context and other processes all affect the observed peak. This is a placement budget, so an exact device-wide free-memory guarantee requires a separate admission mechanism.

The canonical #125 package is 15,548,157,542 bytes. One sequence, context 8,192, FP16 KV, PDL on, speculation off. For each GPU we replayed its 1,024-token README input and fixed 32-token Basalt history. Each of five cycles tested 0/768/1024 MiB in a rotated order and its reverse, for ten fresh processes per setting. Timings cover the first request and exclude model load. The peak is the highest process-memory observation across load and execution at a 4 ms sampling interval.

| GPU | Margin MiB | Fully resident layers / 24 | Prefill width | Prefill ms | Decode 32 ms | Peak GPU bytes | Changed top-1 vs 0 / 33 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5070Ti | 0 | 20 | 1024 | 160.652 | 304.031 | 16,546,529,280 | 0 |
| 5070Ti | 768 | 18 | 1024 | 198.614 | 358.713 | 15,741,222,912 | 0 |
| 5070Ti | 1024 | 17 | 1024 | 198.842 | 385.934 | 15,470,690,304 | 0 |
| 5060Ti | 0 | 20 | 1024 | 380.419 | 747.839 | 16,554,917,888 | 0 |
| 5060Ti | 768 | 18 | 1024 | 492.974 | 949.060 | 15,749,611,520 | 0 |
| 5060Ti | 1024 | 17 | 1024 | 530.693 | 1049.646 | 15,479,078,912 | 0 |

With a 1,024 MiB placement margin, the RTX 5070 Ti reduced its sampled peak by 1,075,838,976 bytes and took 38.190 ms longer to prefill and 81.903 ms longer to decode. Its sampled peak was 50,331,648 bytes below the separate Basalt run using the same history; those timings were 450.1 ms prefill and 635.9 ms decode. These are fixed-input placement results. The top-1 check covers 33 predictions.

On RTX 5060 Ti, the same 1,024 MiB setting saved 1,075,838,976 sampled GPU bytes. Prefill took 150.274 ms longer and the 32-token decode took 301.806 ms longer. All 33 fixed-history predictions agreed at each margin on each card. A zero margin remains the speed-oriented default.

[Raw per-process measurements](inkling-vram-margin.json) · [Basalt comparison and recorded continuation](inkling-basalt-canonical.md). The measured implementation was `56cc658` with cubin `48b7ec2`. Reproduce using the [fixed-token replay driver](../../crates/leansonic-core/examples/inkling_basalt_replay.rs) and the same canonical model and Basalt-generated continuation. Run the following once per margin and repeat in mirrored cycle order.

```sh
LEANSONIC_VRAM_RESERVE_MIB="$MIB" LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_CUBIN_PATH="$CUBIN" \
  "$REPLAY" "$CANONICAL" docs/benchmarks/inputs/inkling-readme1024.ids "$CONTINUATION_IDS"
```
