# Stop tokens and generation limits in speculative decode

On RTX 5070 Ti, the seven-level Inkling target and official DSpark drafter ran with context 2,048, FP16 KV and a single request. The baseline used scalar greedy decode. For each of ten cases, the driver checked callback IDs against the emitted vector, saved the full terminal vocabulary row, fed the returned ID, advanced three scalar steps and saved another row. Each row has 200,058 finite FP32 values (800,232 bytes). The same scalar runner was used on the corrected build.

| Case | Emitted | Scalar returned ID | Previous speculative returned ID | Previous row argmax | Previous resume IDs matched scalar |
|---|---:|---:|---:|---:|---|
| immediate | 0 | 90 | 90 | 90 | yes |
| first-draft | 1 | 198 | 198 | 220 | yes |
| mid-draft | 2 | 220 | 220 | 196980 | yes |
| eos | 82 | 200006 | 200006 | 200006 | no |
| near-eos | 3 | 200006 | 200006 | 200006 | no |
| cap1 | 1 | 90 | 90 | 90 | yes |
| cap2 | 2 | 198 | 90 | 220 | no |
| cap3 | 3 | 220 | 90 | 196980 | no |
| cap7 | 7 | 1 | 3 | 25 | yes |
| fixed | 7 | 25 | 25 | 25 | yes |

Seven cases exposed a difference in the prior build. In the natural EOS case the old returned ID and argmax both equaled 200006, yet the full row and resumed sequence differed. After the fix, all ten cases matched scalar for the emitted IDs, terminal ID and argmax, all three resumed IDs and both full row digests. The fixed-length control requests the successor of the last emitted token. With a stop configured, reaching the cap leaves the last emitted prediction available to the caller.

## Reproduce

Use the same seven-level target, official drafter and cubin as the [parent control](../). Build the example at 1e6585e for the previous speculative path and ace18b3 for the corrected and scalar paths. Copy the two binaries into the fixture directory under the names shown. The driver creates new output directories for each run.

```sh
cp "$RECIPE/stop.rs" crates/leansonic-core/examples/spec_eos.rs
cargo build --offline --release -p leansonic-core --example spec_eos
# In a fresh directory with the two built executables and the three ID fixtures:
cp "$RECIPE/ja_end.ids" "$RECIPE/../ja_reasoning.ids" "$RECIPE/../q11_json_schema.ids" .
export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1
export LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1
export LEANSONIC_CUBIN_PATH=/path/to/engine.cubin
./after "$MODEL" 0 scalar-rows > scalar.csv
./before "$MODEL" 7 before-rows > before.csv
./after "$MODEL" 7 after-rows > after.csv
python3 "$RECIPE/check.py"
```

[Full sequences, SHA-256 hashes and measured revisions](verified.json).
