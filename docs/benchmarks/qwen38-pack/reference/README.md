# Qwen3.8-Flash-Next BF16 quality reference

The reference generator uses the text weights and forward pass from the official Hugging Face checkpoint at revision `812d96e10a0dd82b69d749e68693698cf6c36daa`. The original 131 safetensors shards occupy 360,000,192,888 bytes. Download the index, configuration and tokenizer from the same revision. [Pinned upstream checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa)

The runner instantiates the upstream model on the meta device, copies the dense text weights to CUDA, then reads the chosen expert matrices and Engram embedding rows from the original BF16 shards. It uses the upstream eager expert function in ascending expert order. The sharded lookup follows the numerical partition order. This permits reference evaluation on a single card. Performance of the reference runner depends on storage and host-to-device traffic.

Use an empty output directory name and an idle CUDA GPU. The version set on the RTX PRO 5000 72 GB was PyTorch 2.14.1, Transformers 5.19.0, Accelerate 1.12.0, safetensors 0.8.0 and NumPy 2.3.5. The four inputs cover English, Japanese, code and a repeated phrase. Full logits are accumulated in FP32 files from the BF16 model output. Each sequence uses a causal prefill to obtain predictions at every input position, then a cached decode for the eight greedy tokens. The full score compares 49 next-token distributions across 248,320 vocabulary entries in FP64.

```sh
d=docs/benchmarks/qwen38-pack
s=docs/benchmarks/qwen38-engine-comparison
python3 "$d/reference/bf16_teacher.py" "$OFFICIAL_CHECKPOINT" "$d/fixtures.ids" bf16-score
OPENBLAS_NUM_THREADS=1 python3 "$d/score.py" bf16-score direct-score
python3 "$s/kld_fixture.py" "$d/fixtures.ids" bf16-score four-fixtures.kld
target/release/leansonic eval-kld --weights "$DIRECT" --ref four-fixtures.kld
```

The compact llama.cpp-format reference truncates each input to ten tokens and exports positions 5–8, for 16 target distributions. It encodes log probabilities as uint16. LeanSonic's native scorer and the Basalt PPL gate apply a −16 log-probability cutoff to KL. Compare those scores on the same 16 positions and use the separate 49-position report for full-vocabulary KL.

## Measured quality

On the four fixtures, the Wave 1 Q4 model matched the BF16 top token at 39/49 positions (79.59%). Full forward KL averaged 0.281135 with a maximum of 1.469437. Target-token PPL was 28.0382 for BF16 and 52.9910 for Q4. None of the four eight-token continuations matched in full. The largest KL was after input token 1710 at zero-based position 8 of the Japanese fixture. Both models predicted a newline as their most probable token there, but assigned different probabilities to the rest of the vocabulary. Raw and packed Q4 logits were identical. [49-position summary](bf16-wave1-quality.json) · [Individual positions](bf16-wave1-rows.json)

The compact reference is 7,946,548 bytes with SHA-256 `1138516f0ffa3779c6b66633b0b56f58640310f8d210eb364a6dbf167686c7ae`. The maximum absolute readback error was 0.000240 in log-probability units and every exported row retained its original argmax. Refer to the manifest for the exact target positions and tail-mass bound. [Reference file](qwen38-bf16-four-fixtures.kld) · [Manifest](manifest.json)

| Native scorer, same 16 targets | Mean / worst KL with cutoff | Top-1 | Candidate PPL | BF16 PPL |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic Wave 1, direct Q4, PRO 5000 | 0.30440 / 1.44033 | 13/16 | 13.0983 | 8.4605 |
| Basalt v0.5.7, UD-Q4_K_XL, fixed placement, PRO 4500 | 0.03160 / 0.14095 | 15/16 | 9.5383 | 8.4605 |

Basalt's fixed-placement Q4 output was closer to BF16 on these targets. The engines used different quantization formats. Basalt's adaptive placement has separate performance measurements and can produce different tokens. [LeanSonic native output](native-bf16-cli.json) · [Basalt native output and individual rows](basalt-bf16-ppl.json)
