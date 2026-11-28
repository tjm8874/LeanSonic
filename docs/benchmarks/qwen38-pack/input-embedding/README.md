# Input embedding precision on an unchanged Q4 decoder

The stock input embedding differed by 8.6% in relative L2 from BF16 at the first token of an earlier Japanese trace. This experiment replaces only the target token lookup, for the entire 248,320-token vocabulary. Both serial and batched prefill kernels populate the same four FP32 residual streams. All 48 decoder layers, the Engram and the final projection use the original MLX Q4 model. The Q8 arm stores signed bytes with one BF16 scale per group of 32. Neither arm uses calibration prompts.

On the RTX PRO 5000 72 GB, the unmodified control reproduced the published Q4 logits byte-for-byte at every position in both sets. Each candidate used the same token IDs as the official BF16 teacher. The corresponding four greedy eight-token continuations did not match BF16 in full in either set for any arm.

| Set | Input embedding | Mean / worst KL vs official BF16 | Top-1 | PPL |
| --- | --- | ---: | ---: | ---: |
| Original, 49 positions | Q4 | 0.281135 / 1.469437 | 39/49 | 52.9910 |
| | Q8 | 0.278479 / 1.474271 | 39/49 | 50.3644 |
| | BF16 | 0.276211 / 1.441874 | 40/49 | 49.3464 |
| Held-out, 65 positions | Q4 | 0.369378 / 1.407895 | 44/65 | 25.2686 |
| | Q8 | 0.340043 / 1.235803 | 47/65 | 25.3493 |
| | BF16 | 0.343055 / 1.241303 | 46/65 | 25.0987 |

BF16 reference PPL was 28.0382 on the original set and 19.9354 on held-out prompts. Using the full BF16 input table improved the four reported aggregate measures on both sets; the rest of the Q4 engine still differs substantially from the reference. Q8 improved held-out KL and top-1 while raising its PPL. This is a research comparison on eight inputs. [All quality measures, hashes and size readback](pro5000.json)

The two probe packages include the original Q4 embedding. The original direct checkpoint occupies 110,833,479,038 bytes. The complete Q8 test package is 111,508,909,867 bytes and the complete BF16 package is 112,104,877,655 bytes. They allocate an additional 675,430,400 and 1,271,398,400 pinned host bytes respectively for token lookup. The existing Q4 export remains available to other interfaces. Speculative decoding has not been validated with these probes.

## Stock-format refits and a second GPU

The fixed-size input refit keeps the original group-32 Q4 layout and BF16 affine parameter shapes, initializes from stock and retains the lowest squared weight error found within five iterations. A second refit applies this procedure to all 48 shared experts (gate, up, down and shared expert gate projections); routed experts remain unchanged. Both Q4-only candidates use the stock engine. The temporary overlay keeps the old tensors physically on disk. Replacing them in a release would preserve tensor payload length. The combined arm also uses the BF16 input probe.

| Candidate | Original mean / worst KL | Original top-1 / PPL | Held-out mean / worst KL | Held-out top-1 / PPL |
| --- | ---: | ---: | ---: | ---: |
| Stock | 0.281135 / 1.469437 | 39/49 / 52.9910 | 0.369378 / 1.407895 | 44/65 / 25.2686 |
| Input Q4 refit | 0.265270 / 1.441949 | 39/49 / 52.7949 | 0.363156 / 1.355348 | 45/65 / 25.3341 |
| Shared expert Q4 refit | 0.27021 / 1.30898 | 39/49 / 50.1171 | 0.35405 / 1.04462 | 46/65 / 25.4949 |
| Shared refit + BF16 input | 0.25446 / 1.29002 | 42/49 / 49.4414 | 0.33519 / 1.12719 | 46/65 / 25.4662 |

All three refits lowered mean and worst KL on both input sets, while held-out PPL rose. Keep them experimental. For additional inputs, this branch freezes [sixteen passages](extended.txt) and [their 510 input tokens](extended.ids) covering English, Japanese, code and arithmetic. They supply 494 next-token targets.

The PRO 4500 32 GB independently reproduced all 65 held-out logits and all four continuations from the PRO 5000 for each of the BF16 and Q8 input arms. With the standard 16/128-input, 32-output benchmark and 19 layers in VRAM, the BF16 input had a sampled process peak of 32,040,288,256 bytes, equal to the Q4 control. For the 128-token prompt, Q4 reached 66.35 prompt / 21.79 decode tok/s and BF16 input 63.51 / 20.05. These are complete greedy requests. Changing embedding precision can change output tokens and routing. [Refit scores and all PRO 4500 runs](refits-and-cross32.json)

To build the same-size input refit, use `refit_q4` as the last argument of the builder below. For the shared expert refit use `python3 "$d/input-embedding/shared_experts.py" "$Q4" "$OFFICIAL" "$NEW_SHARED"`. Test each with the native teacher and both sets of prompts.

## Reproduce

Use engine commit `5e236dc` and a directly loadable Q4 and official BF16 checkpoint as described in the [reference setup](../reference/README.md). The builder uses hard links in a new output directory on the same filesystem as the base. The new probe kernels require the matching cubin.

```sh
d=docs/benchmarks/qwen38-pack
python3 "$d/input-embedding/build.py" "$Q4" "$OFFICIAL" "$NEW_Q8" q8
python3 "$d/input-embedding/build.py" "$Q4" "$OFFICIAL" "$NEW_BF16" bf16
mkdir -p crates/leansonic-core/examples
cp "$d/teacher.rs" crates/leansonic-core/examples/q38_teacher.rs
cargo build --release -p leansonic-core --example q38_teacher
nvcc -arch=sm_120a -O3 -std=c++17 -Ikernels --cubin kernels/qwen38_sm120.cu -o probe.cubin
export LEANSONIC_QWEN38_CUBIN_PATH=$PWD/probe.cubin
tail -n 4 docs/benchmarks/qwen38-engine-comparison/routing/route-fixtures.ids > heldout.ids
target/release/examples/q38_teacher "$NEW_BF16" heldout.ids candidate
python3 "$d/score.py" "$HELDOUT_BF16_TEACHER" candidate
```

Repeat with the Q4 control and Q8 candidate and use `$d/fixtures.ids` for the original four prompts. The reference dump must use the identical input IDs.

## Sixteen additional prompts on the RTX PRO 5000

All seven arms used the same 494 next-token positions, the full 248,320-token vocabulary and the official BF16 reference (PPL 11.3821). Prompts and IDs were frozen at `492b72b`. These are authored passages, four per topic. They do not measure downstream task accuracy.

| Candidate | Mean KL | Worst KL | Top-1 | PPL |
| --- | ---: | ---: | ---: | ---: |
| Stock Q4 | 0.269670 | 2.499328 | 406/494 | 15.9460 |
| BF16 input | 0.272190 | 2.441977 | 408/494 | 15.7419 |
| Q8 input | 0.273126 | 2.396028 | 404/494 | 15.7873 |
| Input Q4 refit | 0.271693 | 2.106243 | 408/494 | 16.0659 |
| Shared expert Q4 refit | 0.269370 | 2.415494 | 410/494 | 15.7168 |
| Shared refit + BF16 input | 0.266433 | 2.696336 | 413/494 | 15.7210 |
| Output head Q4 refit | 0.264810 | 2.748131 | 404/494 | 15.9066 |

The two higher-precision input tables raised mean KL on the expanded set. Refitting the output head alone produced the lowest mean KL and raised worst KL from 2.4993 to 2.7481. Both maxima occur at token position 20 of the Python passage (zero-based case 8). The shared-expert-only refit improved all four aggregate measures relative to stock here, although its PPL increased on the previous 65 positions. The stock, BF16 input, Q8 input and input-refit arms matched 1/16 full BF16 continuations. Both shared-expert refits matched 2/16, and the output-head refit matched 3/16. Leave the default model unchanged. [Full scores, worst positions and logit hashes](extended-scores.json)

To reproduce, use `extended.ids` in both the reference and candidate teacher commands above. For the head-only run, pass `refit_head_q4` to `build.py`; it fits the output projection against the BF16 head and preserves the input embedding. This diagnostic overlay measured 111,230,791,349 file bytes, including the superseded Q4 head. Its replacement tensor payload has the same length as the stock head.

## Basalt v0.5.7 on the common 176 positions

Each engine scored positions 11–21 inclusive in each of the same 16 prompts (context 23). KL uses the common uint16 BF16 reference and the −16 log-probability cutoff. The reference PPL on this subset is 7.5174. Basalt uses UD-Q4_K_XL; LeanSonic uses MLX Q4 or its indicated variation. The engine and the weight representation both vary.

| Candidate | Mean KL | Worst KL | Top-1 | PPL |
| --- | ---: | ---: | ---: | ---: |
| stock | 0.259522 | 2.499486 | 156/176 | 10.8242 |
| bf16 | 0.263003 | 2.442311 | 156/176 | 10.7177 |
| q8 | 0.263432 | 2.395195 | 155/176 | 10.7128 |
| shared | 0.260742 | 2.415700 | 153/176 | 10.7108 |
| shared-bf16 | 0.254413 | 2.696595 | 158/176 | 10.5781 |
| basalt-adaptive | 0.016088 | 0.181749 | 170/176 | 7.7054 |
| basalt-fixed | 0.016902 | 0.195750 | 172/176 | 7.6513 |

Both Basalt modes were closer to BF16 on mean and worst KL than all five native variants. To reproduce, use the pinned [engine comparison](../../qwen38-engine-comparison/README.md) and replace its fixtures.ids with extended.ids in a new benchmark directory. Regenerate the BF16 teacher for those same IDs, then run kld_fixture.py, the native eval-kld command and basalt_ppl.py for both configurations. [Native metrics and aligned Basalt rows](extended-public.json)

## Where does the MLX Q4 quality gap appear?

The RTX PRO 5000 ran the same upstream eager model and token IDs with different weight sources. Decoding the MLX routed-expert codes to BF16 when each expert is used preserves the upstream forward pass and tensor orientation. On the four original prompts (49 positions), official BF16 has PPL 28.0382.

| Upstream weights | Mean KL vs BF16 | Worst KL | Top-1 | PPL |
| --- | ---: | ---: | ---: | ---: |
| Only routed experts from MLX Q4 | 0.018161 | 0.127472 | 43/49 | 29.0826 |
| Entire checkpoint from MLX Q4 | 0.284651 | 1.429076 | 39/49 | 53.4910 |

With full MLX Q4 as reference, the native implementation has mean/worst KL 0.021105/0.159408, the same top-1 at 49/49 teacher positions, four matching eight-token continuations out of four, and PPL 52.9910. These 49 positions support investigating quantized weights outside the routed experts. More inputs are required to generalize this result.

On 16 additional prompts (494 positions), the official BF16 engine with only attention V and O weights replaced by MLX Q4 gave mean/worst KL 0.013737/0.149657 for layers 15, 31, 35 and 39, or 0.026107/0.385368 across all 12 attention layers. Respectively 471 and 456 top-1 predictions matched and PPL was 11.3720 and 11.4763 (BF16 11.3821). The unchanged upstream control reproduced all 494 rows and 16 continuations exactly.

The reverse intervention installs original BF16 weights into native Q4 attention. Differences from the previous experiment include the surrounding checkpoint and engine. The unchanged native control also reproduced all prior 494 rows and all 16 continuations byte for byte.

| Native Q4 overlay | Mean KL vs BF16 | Worst KL | Top-1 | PPL | Distribution bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 0.269670 | 2.499328 | 406/494 | 15.9460 | 110,833,479,038 |
| Four layers V+O | 0.265801 | 3.578768 | 408/494 | 16.4132 | 110,969,795,806 |
| Twelve V | 0.267790 | 2.255989 | 408/494 | 15.8504 | 110,864,939,124 |
| Twelve O | 0.253121 | 2.085065 | 408/494 | 16.1145 | 111,210,969,236 |
| Twelve V+O | 0.258190 | 2.614155 | 400/494 | 16.1082 | 111,242,429,338 |

Of 16 full eight-token BF16 continuations, stock, twelve O and twelve V+O each matched one. Four V+O and twelve V each matched two. Adding all V+O tensors cost 408,950,300 file bytes and increased sampled peak process memory from 75,486,986,240 to 75,914,805,248 bytes. The compute-sanitizer projection test reported zero errors and four CPU/guard checks passed. Keep the overlays experimental. [Per-arm scores, logit hashes and streaming measurements](attention-and-upstream.json)

For reproduction, use the upstream Python environment in the reference setup and a fresh output directory for each command. Use the raw MLX checkpoint for the Q4 argument.

```sh
d=docs/benchmarks/qwen38-pack
python3 "$d/input-embedding/upstream_q4.py" "$OFFICIAL" "$Q4" "$d/fixtures.ids" "$OUT/upstream"
python3 "$d/input-embedding/attention_ablation.py" "$OFFICIAL" "$Q4" "$d/input-embedding/extended.ids" "$OUT/official-vo"
python3 "$d/input-embedding/build_attention.py" "$DIRECT" "$OFFICIAL" "$OUT/native-vo" --layers twelve --projections vo
```

Build the native CUDA module and teacher as above on commit `7bdc611`. Run the native teacher on extended.ids with each overlay and score against the frozen BF16 outputs. For separate upstream single-family interventions on the original 49 positions, run quant_families.py with the same arguments as upstream_q4.py. The final restored BF16 arm checks that the original weights have been restored.

## Independent tensor families in the official model

On the RTX PRO 5000 72 GB, quant_families.py replaced one family per run with the MLX Q4 values in an otherwise official BF16 model. Each row below uses the same four original inputs (49 next-token positions). Every official BF16 tensor was restored before the next family. After all six runs, the final control matched the original BF16 FP32 logits byte for byte and matched all four eight-token continuations by ID.

| Quantized family | Mean KL | Worst KL | Top-1 / 49 | PPL |
| --- | ---: | ---: | ---: | ---: |
| Official BF16 control | 0 | 0 | 49 | 28.0382 |
| Hyper-connections | 0.054800 | 0.339982 | 42 | 30.0121 |
| Gated DeltaNet projections | 0.021012 | 0.183284 | 45 | 27.5856 |
| Full-attention projections | 0.021729 | 0.151448 | 44 | 28.8896 |
| Shared experts and their gates | 0.015288 | 0.088391 | 47 | 28.3740 |
| Engram table and two PLE projections | 0.138276 | 2.545091 | 44 | 26.2686 |
| Input embedding and output head | 0.034036 | 0.107086 | 45 | 30.4478 |

Engram produced the highest mean and maximum KL while improving target-token PPL on these inputs. The six KLs measure separate interventions and cannot be summed to predict the combined error. [Full scores and logit hashes](upstream-quant-families.json)

For the table/projection split, run engram_ablation.py with the same four path arguments as upstream_q4.py and choose `--background bf16` or `--background q4`. The latter also replaces the routed experts and dense weights. For a native Q4 hyper-connection refit, run refit_hyperconnection.py with the base package, official checkpoint and a new output directory. Each replacement uses the original group-32 layout and the same tensor payload length. The diagnostic overlay retains the superseded tensors.

The native hyper-connection refit used the 103,556,413,821-byte weight-frame baseline, with a 399,769,600-byte replacement tensor payload. Its diagnostic distribution counted 103,956,318,481 file bytes including the original tensors. On the original 49 positions it lowered mean KL from 0.281135 to 0.268808 and PPL from 52.9910 to 50.2589, while worst KL rose from 1.469437 to 1.604807. On the separate 65 positions, mean/worst KL was 0.364191/1.241401, top-1 44/65 and PPL 24.5753. On the expanded 494 positions, mean/worst KL worsened from 0.269670/2.499328 to 0.279999/2.776550 and PPL increased from 15.9460 to 16.0238, although top-1 improved from 406 to 413. The PRO 5000, 5070 Ti and 5060 Ti produced the same full FP32 output hash and eight-token continuation hash for this refit. Keep the stock hyper-connection quantization. [Scores, streaming trials and GPU hash](hyperconnection-refit.json)

## Split Engram table and projections

Each pair of table and projection precisions was run in both upstream backgrounds with identical input IDs, eager execution and the official BF16 reference. All three controls reproduced the earlier full FP32 logits and continuations exactly. Values use the original 49 teacher positions.

| Other weights | Engram table | PLE key/value projections | Mean KL | Worst KL | Top-1 | PPL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BF16 | BF16 | BF16 | 0 | 0 | 49/49 | 28.0382 |
| BF16 | Q4 | BF16 | 0.009395 | 0.064458 | 47/49 | 28.5413 |
| BF16 | BF16 | Q4 | 0.014053 | 0.122360 | 46/49 | 27.4275 |
| BF16 | Q4 | Q4 | 0.138276 | 2.545091 | 44/49 | 26.2686 |
| MLX Q4 | BF16 | BF16 | 0.298536 | 1.472025 | 39/49 | 53.9897 |
| MLX Q4 | Q4 | BF16 | 0.279530 | 1.495126 | 41/49 | 52.5435 |
| MLX Q4 | BF16 | Q4 | 0.292975 | 1.490800 | 38/49 | 53.6483 |
| MLX Q4 | Q4 | Q4 | 0.284651 | 1.429076 | 39/49 | 53.4910 |

Quantizing both components had a larger effect on the BF16 background than either component in isolation. In the Q4 background, restoring both components worsened mean KL. Restoring the two projections alone lowered mean KL and PPL but raised worst KL. The corresponding native experiment in build_ple.py loads an optional BF16 tensor with 12,800 rows and 2,560 columns. [Raw scores and three exact controls](engram-split.json)

## Native BF16 PLE projections

Commit `b6e7103` registers the fused BF16 PLE projection in the safetensors index. An initial overlay omitted that entry and ran the Q4 control path; it is excluded. For the accepted trials the manifest pointed to the new sidecar and every candidate teacher logged that it loaded BF16 PLE key/value projections. The unchanged control reproduced all 494 native logit rows and all 16 native canaries. Each row gives stock → candidate against the official BF16 teacher on the RTX PRO 5000 72 GB.

| Scored positions | Mean KL | Worst KL | Top-1 matches | PPL |
|---|---:|---:|---:|---:|
| 49 | 0.281135 → 0.276779 | 1.469437 → 1.459692 | 39 → 40 | 52.9910 → 52.6399 |
| 65 | 0.369378 → 0.345537 | 1.407895 → 1.206189 | 44 → 45 | 25.2686 → 24.9235 |
| 494 | 0.269670 → 0.269985 | 2.499328 → 2.046052 | 406 → 408 | 15.9460 → 16.0018 |

The BF16 diagnostic package was 110,899,015,264 bytes including its original Q4 weights, up from 110,833,479,038; sampled peak process VRAM rose by 67,108,864 bytes to 75,554,095,104. On 16/128 prompt tokens with 32 outputs, warm median decode was 163.01/158.96 tok/s for Q4 and 161.93/157.99 for the candidate. Prefill was 397.20/491.68 and 396.38/491.57 tok/s, respectively. Keep the standalone change experimental: worst KL improved in all three sets, but average KL and PPL worsened on the largest. [Full scores, output IDs, per-request times and exact control](native-ple.json)

Build the overlay with `build_ple.py "$DIRECT" "$OFFICIAL" "$NEW"`. Confirm the index maps `language_model.model.layers.1.ple.kv_proj.probe_bf16` to `ple-bf16-probe.safetensors`. Build the teacher and cubin from the same commit, run the three identical fixture files in both arms, and compare to the official teacher with the ID-aware scorer.
