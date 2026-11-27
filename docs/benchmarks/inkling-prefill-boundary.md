# Multi-sequence logits across prefill workspace sizes

A 64-sequence engine can use a 64-row prefill workspace while a single-sequence engine uses 128 or 256 rows. The old prefill loop gave a prompt with 66–95 tokens to different arithmetic paths. A wide workspace processed the entire prompt with Tensor Cores; a narrow workspace processed the first 64 tokens and then used the small-batch path for the final 2–31. Matching the greedy token immediately after prefill missed differences in the subsequent KV-dependent predictions.

The candidate changes the split only when a narrow workspace would leave 2–31 tokens after its final full batch. It reserves 32 tokens for the final call. For example, length 66 becomes 34 + 32 at width 64 and stays a single 66-row call at wider widths. Both calls retain the Tensor Core paths. The existing 65-token schedule and its final single-token decode remain intact. This does not allocate additional workspace or displace expert weights.

A previous candidate split the tail off even on a wide workspace. On RTX 5070 Ti, three mirrored process cycles (18 warm requests per length and arm) measured 95-token prefill at 139.47 ms versus 91.82 ms with the parent. That candidate was set aside. [Per-length timings](https://github.com/tjm8874/LeanSonic/pull/120#issuecomment-7212532430)

## Reproduce on Blackwell

Build the CUDA image and the diagnostic from the same checkout. Use an Inkling-2-Mini text package supported by that checkout and one selected GPU. The program fixes context 256, FP16 KV when set below, C=1 for the independent engine and the specified C for the batched engine. It resets each slot, supplies two fixed continuation IDs, reverses the slot order on the second round, checks for nonfinite logits, and compares all 200,058 FP32 values per row. The CSV gives the slot, step, both top tokens, number of different entries, maximum absolute difference and maximum ULP distance. A difference causes a nonzero exit.

```sh
nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o round.cubin
cargo build --release -p leansonic-core --example inkling_round_logits
export LEANSONIC_CUBIN_PATH="$PWD/round.cubin" LEANSONIC_KV=fp16 LEANSONIC_PDL=1
audit=target/release/examples/inkling_round_logits
ids=docs/benchmarks/inputs/inkling-readme1024.ids
"$audit" "$MODEL" "$ids" 64 64
"$audit" "$MODEL" "$ids" 64 auto
```

The two final arguments are concurrency and the prefill policy. The first command holds both workspace widths at 64; the second uses each engine’s automatic policy. The default prompts cover lengths 32–95. Append `96` or `160` to select the next window of 64 lengths.

On the parent scheduler, the automatic configuration changed 12,003,288 of 25,607,424 logits, including four next-token changes out of 128 positions. The complete per-row CSV was byte-identical on RTX PRO 5000, RTX 5070 Ti and RTX 5060 Ti. Holding both widths at 64 restored equality on the 5070 Ti and 5060 Ti. These fixtures exercise end-to-end prompt processing and decode; a corpus comparison to BF16 is a separate measurement. [Investigation and reproducer](https://github.com/tjm8874/LeanSonic/issues/107#issuecomment-7212247468)

The [BF16 observer, timing driver and instructions](https://github.com/tjm8874/LeanSonic/blob/90eb3b1/docs/benchmarks/inkling-prefill-schedule-probe.md) also allow separate comparisons of schedule-dependent quality and latency.

## Candidate measurements

Automatic prefill width, FP16 KV, PDL, context 256, 64 independent prompts, 2 forced rounds per slot. Each completed window compares 25,607,424 FP32 logits and 128 decode predictions, plus the initial greedy prediction following prefill.

| GPU | Prompt lengths | Unequal FP32 values | Changed first tokens after prefill | Changed decode top-1 |
|---|---|---:|---:|---:|
| PRO4500 | 32–95 | 0 | 0 | 0 |
| PRO4500 | 96–159 | 0 | 0 | 0 |
| 5070Ti | 32–95 | 0 | 0 | 0 |
| 5070Ti | 96–159 | 0 | 0 | 0 |
| 5070Ti | 160–223 | 0 | 0 | 0 |
