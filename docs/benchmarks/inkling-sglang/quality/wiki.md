# SGLang on the WikiText-2 BF16 corpus oracle

Every row is a pre-sampling score from the original SGLang model. Each chunk starts with the exact first 257 IDs of the stored 512-token reference. The other 255 IDs are forced in order. The sampling hook records the logits before selecting each next input. Before forwarding any rows to the same LeanSonic native CPU scorer, the driver verifies that all 966 additional SGLang output positions are masked to negative infinity. Each model receives the same input IDs, KLD oracle and target positions.

| Split | Engine / distribution | Mean KL | Worst KL | Top-1 / 25,500 | PPL |
|---|---|---:|---:|---:|---:|
| test | Canonical LeanSonic | 0.0031584 | 0.6809773 | 24,865 | 7.072956 |
| test | Original SGLang | 0.0103728 | 3.7027736 | 24,375 | 7.084607 |
| test | Original llama.cpp Q3_K_XL | 0.0179920 | 6.3689020 | 24,081 | 7.134621 |

SGLang's test maximum was at chunk 23, token 330. Its p99 KL was 0.0879798.

SGLang uses the pinned original weights and BF16 KV on RTX PRO 5000. The canonical LeanSonic and original Q3 measurements use FP16 KV. Both LeanSonic and SGLang results above apply the identical native scoring code, including its low-probability cutoff. The original GGUF engine used its published batched native evaluator. This instrumentation measures quality. See the [uninstrumented timings](../README.md) for speed and allocation.

## Reproduce

Prepare the two [BF16 oracles](../../../inkling-reference.md) and use the [same SGLang environment](README.md). Use a fresh directory. CLI is the absolute path to a LeanSonic binary with the native eval-kld --logits-f32 option. All measured file hashes are in [wiki-results.json](wiki-results.json).

```sh
quality="$REPO/docs/benchmarks/inkling-sglang/quality"
export PYTHONPATH="$quality:${PYTHONPATH:-}" HF_HUB_OFFLINE=1
cp "$quality/../functional.json" .
INKLING_CAPTURE="$PWD/test.rows" python3 "$quality/wiki.py" "$TEST_KLD" "$CLI" test
INKLING_CAPTURE="$PWD/valid.rows" python3 "$quality/wiki.py" "$VALID_KLD" "$CLI" valid
```
