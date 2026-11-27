# Same-device BF16 fidelity for the three timed packages

All completed measurements below use the RTX PRO 4500, the same WikiText BF16 oracle, FP16 KV and the exact model files used in the [three-engine timing replay](../README.md). Each split contains 100 independent 512-token contexts. Each engine scores the same 255 teacher-forced positions per chunk. KL is measured over the model vocabulary using the native oracle's log-probability cutoff. The subset of completed arms can be smaller than the timing table.

| Split | Engine | Mean KL | Worst KL | BF16 top-1 / 25,500 | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | vllm | 0.0101532 | 3.1885196 | 24,395 | 7.085312 |
| test | lean | 0.0172238 | 6.1395663 | 24,071 | 7.128123 |
| test | llama | 0.0276770 | 4.5351470 | 23,633 | 7.179674 |

LeanSonic uses direct prefill width 512 and both mixed device projections. Native llama.cpp uses `llama-perplexity --kl-divergence --kl-divergence-base ORACLE -f CORPUS -m GGUF -c 512 -b 512 -ub 512 -ctk f16 -ctv f16 -ngl 99 -t 8 --no-warmup --chunks 100`. vLLM uses the same pinned wheel, BF16 activation dtype, FP16 KV, per-request score observation, one sequence and its 2,048 maximum context. Its collector reads FP32 logits before forcing each next oracle token. It verifies the first captured top ID against an unconstrained greedy request and checks the complete generated token history of every chunk. Prefill starts with 257 IDs and the following 255 distributions are scored. Use the same executable for the LeanSonic run and CPU scoring of vLLM logits.

In a fresh directory containing capture.py and wiki.py, set OFFICIAL and PYTHONPATH to the model and script directory, set VLLM_CAPTURE_DIR to the current output directory, and run `python wiki.py ORACLE SCORER test` (or `valid`). For the published row-level checks use the scorer instrumented to emit rows to DIAG_ROWS. The native top-1 count is recovered from its three-decimal percentage. [Complete score fields, oracle hashes, executable identities, and aligned rows at each engine's maximum when both row files are available](scores.json).

To reproduce ordered rows, apply row_scorer.py to crates/leansonic-cli/src/eval_kld.rs in a disposable a1592f3 checkout and build leansonic-cli in release mode. Set DIAG_ROWS to a new file before each invocation.

On test, LeanSonic alone matches BF16 at 584 positions, vLLM alone at 908, both at 23,487, and neither at 521. LeanSonic exceeds the paired vLLM KL by more than one nat at 5 positions, and vLLM exceeds LeanSonic by more than one nat at 2. [Ten largest differences in each direction](test-delta.json).
