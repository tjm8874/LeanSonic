# Inkling-2-Mini router replay

This experiment records the normalized router input and selected experts at the final prompt token. The two investigated positions were selected for their known divergence. Each of the 48 visits has its own row. Expert lists in the CSV give the set difference from the streaming BF16 reference. Relative L2 uses the BF16 input norm as the denominator.

Apply the adjacent patch with `git apply` to Basalt at `5fe1928dcc82ad62286d2a622bcbb5d85ebfba2b`, then rebuild the Inkling CLI. Run on one RTX PRO 4500 Blackwell. Use the same public token IDs and BF16 oracle as [the prefix comparison](../inkling-basalt-oracle-prefixes.json). Use a fresh trace file for every process. The trace appends one little-endian record per visit, consisting of four int32 fields (visit, physical layer, embedding width, expert count), 2,304 FP32 normalized activations and eight int32 expert IDs.

```sh
BASALT_TRACE_FINAL=prefix.route basalt --model Inkling-2-Mini-NVFP4-mxfp8.basalt --tokens-file prefix.ids --gpu 0 --kv fp16 --max-context 8192 --spec 0 --temperature 0 --max-new 1 --diag-logits prefix.logits
```

For the streaming layout, also export `BASALT_REPLAY_STREAM=1`. It processes four 64-token prompt blocks, then consumes each remaining corpus token individually. The trace is enabled only when the CLI starts the final token. For each streaming control, keep this schedule and omit the trace variable. All 200,058 FP32 logit words matched within each trace/control pair. Native controls at 0/256, 0/400, 29/443 and 69/353 matched unmodified upstream Basalt.

The archived LeanSonic routing probe and streaming BF16 captures are described in [the prior router experiment](https://github.com/tjm8874/LeanSonic/issues/35#issuecomment-7201813771). To reproduce the interventions, extract the normalized vector from visit 11 at 69/353 or visit 13 at 29/443. Pass that vector to the same single-visit LeanSonic probe in router-only mode. The probe restores the original activation before evaluating the experts. Both stock controls reproduced their archived logit bytes and current main's reported row KL. Scores use the full vocabulary against the original KLD reference. The streaming BF16 capture supplies the expert sets and activations for the layer comparisons.

| Chunk / position | LeanSonic stock KL | KL with native Basalt router input | KL with streaming Basalt router input | BF16 top token | LeanSonic top token, stock / native input / stream input |
| --- | ---: | ---: | ---: | ---: | --- |
| 69 / 353 | 0.663992 | 0.012323 | 0.005933 | 2952 | 1057 / 1057 / 2952 |
| 29 / 443 | 0.193547 | 0.145719 | 0.147087 | 149918 | 149918 / 149918 / 149918 |

This intervention changes both expert selection and the associated mixing weights. The two prefixes establish a local causal effect. A production change still needs full-corpus testing.

### Replay the expert IDs and mixing weights independently

On RTX PRO 5000, a second probe wrote just eight little-endian int32 IDs and eight FP32 weights after the normal router and before the expert kernels. The stock arm and the replay of the donor's original ordered pairs each reproduced all 200,058 archived logits at both positions and for both donors. At 69/353, replacing only ID 53 with 20 and keeping the stock weights reduced KL to 0.003912 and recovered top token 2952. Keeping the stock IDs and applying the aligned streaming-donor weights raised KL to 0.747460.

At 29/443 both donors replace expert 76 with 12. With the stock weights, KL stayed at 0.149699. Applying the same streaming-donor ID/weight pairs in two orders gave KL 0.180547 and 0.147087. The router input first changed at the next visit (14). The first subsequent change in the expert **set** was at visit 24. The order difference is measurable for this prefix. It does not establish a generally preferable order.

To reproduce, use LeanSonic `63680ee65c2b3a639893722f32b3c86924e3cbe6`, apply `lean-pair.patch`, build the `route_probe` example in release mode, and build the native `sm_120a` cubin from that checkout. Take the indicated corpus token IDs up to and including the scored token. The `factorial.csv` columns record the exact ordered IDs and IEEE-754 weight words read back from the device. Write any row as follows, where `row` is a Python `csv.DictReader` row:

```python
import struct
from pathlib import Path

ids = [int(x) for x in row["expert_ids"].split()]
words = [int(x, 16) for x in row["weight_fp32_hex"].split()]
Path("pair.bin").write_bytes(struct.pack("<8i8I", *ids, *words))
```

```sh
PROBE_SLOT=11 PROBE_PAIR=pair.bin LEANSONIC_CUBIN_PATH=ink2.cubin \
  target/release/examples/route_probe "$MODEL_DIR" prefix69.ids fresh.route row.f32
```

Use slot 13 for chunk 29. The probe appends to the route trace. Use a fresh path. `self` replays stock; `ids` replaces only the displaced expert in its slot; `wts` uses donor weights mapped onto the stock slots, assigning the incoming expert's weight to the displaced expert. `joint` combines the two interventions in stock slot order, and `original` uses the donor's order. A PRO 4500 repeated the published patch at 69/353 (stream, IDs only) and 29/443 (stream, both orders). All three complete route traces and logit files matched the instrumented PRO 5000 runs byte for byte.
