# Router-input sensitivity on two BF16 reference schedules

These are single-position interventions on unmodified V13 weights, FP16 KV, RTX PRO 5000 72GB. The same engine FP32 logit bytes were checked against an independent production CLI dump before changing any router input. Each intervention copies one BF16-engine FP32 normalized activation into scratch for a single router call. The downstream feed-forward activation and other calls remain native. Only the first three visits with a different top-eight expert set in execution order were tested, one at a time.

| Split / zero-based chunk, position | Reference used for routing and scoring | Native KL | Best of three single substitutions | Top token, native → intervention / BF16 |
|---|---|---:|---|---|
| valid 46, 379 | canonical | 0.352436 | 0.104710, slot 26 | 17 → 170558 / 170558 |
| valid 46, 379 | serial | 0.483306 | 0.382601, slot 13 | 17 → 17 / 170558 |
| test 80, 452 | serial | 0.827418 | 0.282827, slot 7 | 197045 → 197045 / 196977 |

Slot 26 is the second visit to physical layer 8. With the serial activation at that same slot on validation, serial KL was 0.418351 and the top token stayed 17. On the test position, tracing the canonical BF16 graph altered its final logit bytes. Canonical traced activations for that row were excluded. Both validation controls and the serial test control were byte-exact. The [complete results](inkling-bf16-router-interventions.json) include both reference scores for every intervention, observer hashes, and mismatching slots. A corrected prediction on one input does not establish a production remedy.

## Reproduction

1. Generate canonical and serial oracles as in [the reference-schedule recipe](inkling-bf16-serial.md). Apply `tools/inkling_bf16_router_observer.patch` to llama.cpp `29b52d3` **after** the serial-schedule patch. Rebuild `llama-perplexity` and its implementation library. Work in a separate diagnostic build directory.
2. Set `REPLAY_CHUNK` and `REPLAY_POSITION` to the zero-based coordinates above. Make an empty `REPLAY_OUT` directory per run. Execute `llama-perplexity -m "$BF16_GGUF" -f "$WIKITEXT" -c 512 -b "$B" -ub "$B" -ctk f16 -ctv f16 -ngl 99 --no-warmup -t 8 --chunks 100`. Use B=8192 and unset `LEANSONIC_BF16_SERIAL_SCORE` for canonical; B=512 and set it to 1 for serial. Run once with `REPLAY_TRACE` unset and once with it set to 1. Compare the two `logits.bin` files with `cmp` before using any traced activations.
3. On LeanSonic `1444305`, apply `tools/inkling_router_input_probe.patch` and run `cargo build --release -p leansonic-core --example route_probe`. Use the V13 cubin, compact model and FP16 KV. To make `row.ids`, seek to `20 + 4 * 512 * chunk` in either oracle file and read `position + 1` little-endian u32 token IDs. Write a comma-separated line with trailing newline. Verify the input SHA against the JSON.
4. Run `target/release/examples/route_probe "$WEIGHTS" row.ids native.route native.f32` in a fresh directory. The probe prefills the first 257 IDs, follows the supplied tokens, and traces the last forward. Compare the 800,232 bytes of FP32 logits to the stock full-vocabulary dump for that row. Trace records contain little-endian layer, slot, position; 8 expert IDs, 8 FP32 gate weights, 2,304 FP32 router inputs and 2,304 FP32 hidden values. Execution slot order is 0–17, 24–47, 18–23.
5. Compare the native expert set for each slot to the eight little-endian i32 values in `ffn_moe_topk-SLOT.0.bin`. With exact observer controls, rerun the example with `PROBE_SLOT=SLOT` and `PROBE_NORM=/absolute/path/to/norm-SLOT.0.bin`, passing fresh output filenames per trial. Use the same reference-row decoder and KL function as `tools/compare_inkling_oracle_rows.py` to score the resulting 200,058 logits.
