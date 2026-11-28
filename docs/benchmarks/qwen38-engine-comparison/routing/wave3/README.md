# Profile-guided Z24 expert placement on Wave 3

On an RTX PRO 4500 Blackwell 32 GB, selecting individual routed experts for VRAM increased decode throughput from 21.8 tok/s to 30.3–68.7 tok/s over four held-out inputs. The sampled process peak rose from 29.840 to 29.881 GiB. All 80 native 32-token generations, including warmups, matched the stock build for their respective inputs. On the same GPU, Basalt v0.5.7 adaptive placement decoded faster on each of the four cases. Its UD-Q4_K_XL quantization differs from the MLX Q4 used by LeanSonic.

The stock engine fills 19 complete layers and consults host memory for the remaining 29. The experimental engine uses the same 9,728 GPU expert records, spreading them across all 48 layers. The first 32 layers get 203 slots each and the remaining 16 get 202. Routed projections use three device pointer tables per layer. Each pointer leads to a 972,800-byte record in VRAM or mapped pinned host memory, holding the original Q4 weights and losslessly packed Z24 scale and bias values. The host and device each store their own subset of experts. The profile uses the most frequently selected experts per layer from the first four routing calibration prompts of the earlier v0.11 trace. Selection used none of the four timing prompts. The fixed random profile controls for layout and table overhead.

## Timings

Each cell gives prompt tok/s through receipt of the first token, followed by decode tok/s over the remaining 31 output intervals. Each launch warmed all four inputs once, then measured each input three times. Native runs followed stock, calibrated, random, calibrated, stock. The public Basalt binary then ran one adaptive process and one fixed process. Results are medians within each arm. Context was 256, greedy output length was 32, and prompt reuse was disabled. Peak VRAM was sampled per process at 25 ms intervals.

| Input | Stock | Calibrated | Random | Basalt adaptive | Basalt fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| English, 19 tokens | 50.20 / 21.84 | 113.81 / 68.71 | 51.56 / 23.83 | 93.71 / 92.44 | 62.24 / 75.42 |
| Japanese, 25 tokens | 59.56 / 21.84 | 77.46 / 58.92 | 60.27 / 23.80 | 88.27 / 92.10 | 46.12 / 62.84 |
| Code, 6 tokens | 37.20 / 21.85 | 49.40 / 30.33 | 38.26 / 24.05 | 81.71 / 78.55 | 40.83 / 41.15 |
| Arithmetic, 19 tokens | 54.52 / 21.84 | 111.58 / 43.32 | 57.09 / 24.37 | 93.61 / 87.84 | 57.05 / 57.90 |
| Peak GiB | 29.840 | 29.881 | 29.881 | 30.592 | 30.590 |

The directly loadable LeanSonic model occupies 110,833,479,038 bytes. The optional calibrated profile adds 36,500 bytes. Basalt's model file occupies 115,307,561,176 bytes. Raw per-launch timings, generated IDs and teacher scores are in [pro4500.json](pro4500.json). Public Basalt binary and checkpoint hashes and all server flags are in the [engine comparison](../../README.md).

The calibrated layout reproduced every FP32 logit at the original 49 teacher positions and at all 65 held-out positions. All eight eight-token continuations across these two sets matched stock Q4. The random layout also matched at all 49 original positions. Against official BF16 on the held-out set, the calibrated Q4 result had mean KL 0.369378, worst KL 1.407895, top-1 agreement 44/65 and PPL 25.2686 versus BF16's 19.9354. None of the four complete held-out continuations matched BF16. Retain this placement as an opt-in experiment while testing broader distributions and memory budgets.

## Reproduction

Use a scratch checkout of the experimental branch. The GPU comparison used commit `bd62828` for mixed records and stock commit `d46621d`. With the directly loadable Q4 model available as `$MODEL` and the selected GPU's UUID in `CUDA_VISIBLE_DEVICES`, build and run:

```sh
d=docs/benchmarks/qwen38-engine-comparison
p=$d/routing/wave3
mkdir -p crates/leansonic-core/examples
cp "$d/q38_stream.rs" crates/leansonic-core/examples/
cp docs/benchmarks/qwen38-pack/teacher.rs crates/leansonic-core/examples/q38_teacher.rs
cargo build --release -p leansonic-core --example q38_stream --example q38_teacher
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels -DLEANSONIC_Q38_MIXED --cubin kernels/qwen38_sm120.cu -o mixed.cubin
export LEANSONIC_QWEN38_CUBIN_PATH=$PWD/mixed.cubin
export Q38_EXPERT_PROFILE=$PWD/$p/profile-calibrated-19layers.txt
python3 "$d/compare_stream.py" --held-out-routing "$d/routing/route-fixtures.ids" calibrated.log calibrated.json target/release/examples/q38_stream "$MODEL" 256
target/release/examples/q38_teacher "$MODEL" docs/benchmarks/qwen38-pack/fixtures.ids teacher-calibrated
```

Use the second profile for the random control. For stock, use the ordinary cubin with the profile unset and `LEANSONIC_MAX_RESIDENT_LAYERS=19`. For Basalt, use the same four inputs and the two documented sampler/placement configurations. To regenerate the two profiles, follow the [routing trace instructions](../README.md), then run `make_profile.py --resident-layers 19`. A set of 48 profile rows is specific to this model and the selected memory budget. Automatic resizing for long contexts and MTP is outside this experiment.

## Matched BF16 quality gates on the held-out prompts

LeanSonic and Basalt v0.5.7 scored the same four official BF16 references on the PRO 4500. Each prompt used its own full-length context (19, 25, 6 and 19). Both native scorers evaluate next-token targets from position `context // 2` through `context - 2`, for 32 positions in total. They use a uint16 log-probability reference and omit terms with reference log probability at or below −16 from the KL sum. Top-1 and the target-token log losses use the same target positions. Each engine started a fresh process per prompt for quality scoring. The timing runs above kept their processes resident across prompts. The original and calibrated LeanSonic layouts produced byte-identical distributions at all 65 available positions, so the stock native CLI was used for this common subset.

| Candidate | Mean / worst KL vs BF16 | Top-1 agreement | Target PPL |
| --- | ---: | ---: | ---: |
| LeanSonic MLX Q4 | 0.37138 / 1.39348 | 20/32 | 16.5712 |
| Basalt adaptive UD-Q4_K_XL | 0.02914 / 0.11017 | 28/32 | 11.0135 |
| Basalt fixed UD-Q4_K_XL | 0.02843 / 0.11843 | 29/32 | 11.1264 |
| Official BF16 | — | 32/32 | 11.6129 |

PPL measures probability assigned to the observed next tokens in this small sample. KL measures how closely the candidate matches the BF16 distribution. Both Basalt configurations were closer to BF16 here. [Per-case native scores and combined results](common-kld.json)

To reproduce, first produce the BF16 teacher dump with the [official reference runner](../../../qwen38-pack/reference/README.md) on the last four inputs of `route-fixtures.ids`. Point `PYTHONPATH` at the absolute engine-comparison directory. Run [split_reference.py](split_reference.py) with the four input lines, BF16 teacher output directory, and a new destination. In each numbered subdirectory run `leansonic eval-kld --weights "$MODEL" --ref ref.kld`. For Basalt, run `basalt_ppl.py "$PWD" "$PWD/score.json" "$BASALT"` followed by the same server and sampler flags as the fixed or adaptive timing run. Pool NLL and KL over scored positions before computing combined PPL and mean KL.

The released CUDA driver's runtime compiler independently reproduced the reference Q4 logits and eight-token continuations on all 65 held-out positions on the 72 GB PRO 5000. The four new sm_120a kernels also passed the tested exact-size boundary at expert 511 on both device and mapped host records with 65,536 bytes of Compute Sanitizer padding. [Cross-device hashes and sanitizer output](validation.json)
