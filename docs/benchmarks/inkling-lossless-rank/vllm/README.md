# Four-bank Inkling-2-Mini versus publisher vLLM on RTX PRO 4500

LeanSonic 9bc4d48 uses the lossless ranked embedding and four expert banks. vLLM 8c17e469 uses the publisher model b85c3cfa. Both use FP16 KV, a 2,048-token context, one sequence, 1,024 identical input tokens, and the same 32-token decode history. Each phase produces the next token. The LeanSonic CLI reports all 24 expert layers resident. GPU driver 595.99.02 and 200 W power limit.

**2 of five mirrored cycles complete**, with 28 warm requests per engine. Each cycle uses four fresh processes. The initial request of every process is excluded. Speeds are the token counts divided by the mean phase duration. Peak GPU bytes are sampled over each process tree, including model loading. vLLM reserves 2,415,919,104 bytes for its KV cache.

| Engine | Prefill tok/s | Decode tok/s | Peak GPU bytes | Distribution bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic rank | 8,639.61 | 123.68 | 15,590,227,968 | 13,438,517,744 |
| vLLM native | 8,660.98 | 107.65 | 22,756,196,352 | 19,555,785,477 |

The publisher count includes the official tokenizer and configuration and excludes the optional 635,801,416-byte MTP file. The LeanSonic package is the shard and tokenizer in the [physical package receipt](../receipt.json). Each engine disables speculative decoding. vLLM records scheduled-to-first-token and first-to-last-token times. LeanSonic records synchronous in-process phase times.

## Quality

The normal LeanSonic CLI was rebuilt at the timed source revision. vLLM scores come from its prior [same-card full-corpus run](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines/quality), against the same BF16 oracle. Each complete split covers 100 chunks of 512 tokens, scoring 25,500 positions.

| Split | Engine | Mean KL | Worst KL | BF16 top-1 matches | PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| test | LeanSonic rank | 0.0077481 | 1.4620404 | 24,537 | 7.083731 |
| test | vLLM native | 0.0101532 | 3.1885196 | 24,395 | 7.085312 |
| valid | LeanSonic rank | 0.0069527 | 0.7446714 | 24,645 | 6.530797 |
| valid | vLLM native | 0.0092906 | 2.8943274 | 24,490 | 6.533192 |

On test, rank has lower mean KL in 92/100 chunks. A paired bootstrap over whole chunks (10,000 draws) gives a 95% interval of [-0.0028989, -0.0019751] for rank minus vLLM mean KL, [0.286, 0.835] percentage points for top-1, and [-0.262%, 0.216%] for relative PPL. [Largest row differences and paired results](paired-test.json). Regenerate with [paired_quality.py](paired_quality.py) on rank then vLLM score rows.

On valid, rank has lower mean KL in 93/100 chunks. A paired bootstrap over whole chunks (10,000 draws) gives a 95% interval of [-0.0027786, -0.0019346] for rank minus vLLM mean KL, [0.349, 0.871] percentage points for top-1, and [-0.265%, 0.192%] for relative PPL. [Largest row differences and paired results](paired-valid.json). Regenerate with [paired_quality.py](paired_quality.py) on rank then vLLM score rows.

## Reproduce

Build the release leansonic-core example [lean.rs](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines/lean.rs) as bench at 9bc4d48. Build its SM120a cubin as described in the [GPU replay](../README.md#gpu-replay). Use [native.py, peak.py, prompt.ids, anchor.ids and the pinned vLLM runtime](https://github.com/tjm8874/LeanSonic/tree/4d40691/docs/benchmarks/inkling-class-frames/three-engines). Use the [four-bank leg runner](leg.py) from this directory. Save the artifacts under their original filenames and set OFFICIAL, LEAN_MODEL, VLLM_PYTHON and LEANSONIC_CUBIN_PATH. Select the test GPU by UUID. Enable LEANSONIC_PDL, LEANSONIC_PDL_ALL, LEANSONIC_MXFP8_DEVICE_U4 and LEANSONIC_MXFP8_DIRECT_ALL. Set LEANSONIC_KV=fp16, LEANSONIC_CTX=2048, LEANSONIC_SEQS=1 and LEANSONIC_PREFILL_CHUNK=1024. Run in a new directory:

```sh
mkdir runs
for c in 1 2 3 4 5; do
    case "$c" in
        1|3|5) arms="lean vllm vllm lean" ;;
        2|4) arms="vllm lean lean vllm" ;;
    esac
    leg=0
    for arm in $arms; do
        python3 leg.py "runs/$c-$leg-$arm" "$arm"
        leg=$((leg+1))
    done
done
```

For quality, set LEANSONIC_CTX=8192 and LEANSONIC_PREFILL_CHUNK=512. Use [observe.py](../observe.py) to emit DIAG_ROWS for each test/validation oracle, then run `leansonic eval-kld --weights MODEL --ref ORACLE --chunks 100`. The receipt includes the reference, row, executable and cubin hashes, phase times, output IDs and sampled peaks. [Compressed measurement JSON](measurement.json.gz).
