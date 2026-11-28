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
