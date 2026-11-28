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

The two higher-precision input tables raised mean KL on the expanded set. Refitting the output head alone produced the lowest mean KL and raised worst KL from 2.4993 to 2.7481. Both maxima occur at token position 20 of the Python passage (zero-based case 8). The shared-expert-only refit improved all four aggregate measures relative to stock here, although its PPL increased on the previous 65 positions. None of the seven arms reproduced any full eight-token BF16 continuation on these 16 prompts. Leave the default model unchanged. [Full scores, worst positions and logit hashes](extended-scores.json)

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
