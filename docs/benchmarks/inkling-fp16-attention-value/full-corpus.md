# Full BF16 replay and rounding tradeoffs

Candidate ea3a7cf and the current recurrence were each scored on the same 571-slice model and BF16 references. The candidate's complete row files, including full-vocabulary fingerprints, are byte-identical on RTX PRO 5000 and RTX 5060 Ti for both splits. They independently match the earlier G1-only run on RTX 5070 Ti. Each split has 100 WikiText chunks and 25,500 scored positions.

| Split | Recurrence | Mean KL | Worst KL | BF16 top-1 | PPL | 99th percentile KL |
|---|---|---:|---:|---:|---:|---:|
| test | current | 0.0172238 | 6.1395663 | 24,071 (94.40%) | 7.128123 | 0.1486843 |
| test | ea3a7cf | 0.0169571 | 2.8585658 | 24,084 (94.45%) | 7.127375 | 0.1531903 |
| valid | current | 0.0157144 | 2.6382777 | 24,192 (94.87%) | 6.565670 | 0.1472817 |
| valid | ea3a7cf | 0.0157347 | 2.5524491 | 24,200 (94.90%) | 6.566551 | 0.1525733 |

The candidate recovers 215 BF16 top tokens and loses 202 on test, and recovers 203 and loses 195 on validation. The large improvement on test chunk 2 position 478 coexists with a regression on test 92:387 (KL 0.068371 to 0.919946 with unchanged top token). Validation mean KL and PPL rise slightly. Both splits improve worst KL and top-1 while increasing p99 KL. This tradeoff keeps the default change under review.

On PRO 5000 both dispatch pair tests pass 704 cases each. The current operator differs from an independent FP64 reference on zero of 202,752 Q8 values; the proposed operator differs on two. The candidate's entire FP64 fixture CSV is identical on both GPUs. [Baseline FP64 fixtures](current-operator-pro5000.csv) · [Candidate FP64 fixtures](candidate-operator-pro5000.csv) · [Model, oracle, binary and row hashes, all scores and tails](candidate-pro5000.json) · [5060 Ti evidence](candidate-5060.json).

To reproduce, use the 571-slice model digest and the pinned scorer in the [isolation instructions](README.md). Score test and validation separately with all 100 chunks, a 512-token prefill chunk, one sequence, FP16 KV, spec off and both mixed-even device paths enabled. Confirm 24/24 resident layers and 16 mixed-even planes. Set DIAG_ROWS separately for each run. Compare current, then candidate on each device and use sha256sum on each complete row file. The full corpus uses G1; the explicit fallback dispatch was also checked on test chunk 2.

## Independent single-round alternative

For finite online-softmax scores the previous maximum or the new score equals the updated maximum. One of the two exponential weights is therefore one. Fusing the other weight's product with the unscaled term rounds the entire weighted sum once. The separate branch implementation uses one exponential per head and iteration of the G1 loop. It can run on the existing FP32 execution units. This observation concerns one recurrence, while full attention also depends on score reduction, exponent approximation and quantization.

The [scalar check](single-round/scalar.py) compares each FP32 result with an exact rational sum rounded to the nearest FP32 value with ties to even. Across 49,152 deterministic inputs, current and ea3a7cf miss the scalar reference on 501 and 417 inputs respectively. The single-round version matches all inputs. An initial CUDA formulation had an 8-byte stack and was rejected. The [revised patch](single-round/once.patch) on top of ea3a7cf passes the SM120a SASS audit. G1 uses 64 registers and zero stack. Only the three FP16 attention functions changed among 243 symbols. Full operator, corpus and throughput checks are required before choosing a default. [Scalar results](single-round/scalar.json) · [Binary and disassembly evidence](single-round/build.json).


## RTX 5070 Ti runtime

At the same 1,024 prompt tokens and 32 teacher-forced decode inputs, current and ea3a7cf respectively measured **8,257.00 and 8,256.62 prefill tok/s**, **172.369 and 172.382 decode tok/s**, and **14,784,921,600 bytes** peak on both. Five matched ABBA cycles (20 processes, 40 warm requests per arm) gave candidate/current cycle-bootstrap 95% ratios of 0.99962–1.00025 for prefill and 0.99968–1.00055 for decode. The numerical tradeoff is the adoption question on this GPU. [All per-process timings, predictions, peaks and binary hashes](timing-5070.json).

On the independent PRO 5000 check, the follow-up single-round CUDA branch passes 704 batched-vs-staged and 704 staged-vs-grouped cases. Both dispatches produced zero Q8-code mismatches against the CPU FP64 reference across 202,752 codes each. Their block scale bits also agree with each other on every fixture. The maximum absolute scale error against FP64 was 9.79e-10. [All operator fixtures](single-round/operator-pro5000.csv).

## Single-round model replay

On RTX 5070 Ti, the grouped and fallback single-round implementations reproduce the same score/fingerprint file across all 255 test chunk 2 positions. At position 478 KL is **0.003319562**, versus 6.139566282 for current and 0.035403482 for ea3a7cf. The single-round, ea3a7cf and BF16 top token is 44259.

| Split | Recurrence | Mean KL | Worst KL | BF16 top-1 | PPL | p99 KL |
|---|---|---:|---:|---:|---:|---:|
| test | current | 0.0172238 | 6.1395663 | 24,071/25,500 | 7.128123 | 0.1486843 |
| test | ea3a7cf | 0.0169571 | 2.8585658 | 24,084/25,500 | 7.127375 | 0.1531903 |
| test | single-round | 0.0169034 | 2.5614264 | 24,073/25,500 | 7.128507 | 0.1492333 |
| valid | current | 0.0157144 | 2.6382777 | 24,192/25,500 | 6.565670 | 0.1472817 |
| valid | ea3a7cf | 0.0157347 | 2.5524491 | 24,200/25,500 | 6.566551 | 0.1525733 |
| valid | single-round | 0.0155615 | 2.4252143 | 24,208/25,500 | 6.565918 | 0.1496439 |

The current and ea3a7cf controls use the PRO 5000 and 5060 Ti full row files above. The new column uses the 5070 Ti. All three score the same underlying model and use the same oracle and scorer settings. See the [row hashes and paired per-position changes](single-round/candidate-5070.json).

The single-round candidate has the lowest mean and worst KL of these three recurrences on both splits. Relative to current it gains 2 test and 16 validation BF16 top tokens. PPL increases by 0.000383 on test and 0.000248 on validation. Relative to ea3a7cf it loses 11 test top matches while gaining 8 on validation. Its new worst test position is chunk 98 position 465. These results motivate testing the single-round implementation as the draft head. A default or release decision also depends on throughput for this specific implementation and its results on other precisions.
