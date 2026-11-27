# Three-engine Inkling-2-Mini replay on RTX PRO 4500

This measurement uses LeanSonic a1592f3 with its 571-slice split-class package, Unsloth llama.cpp 883286076 with the native IQ3_XXS GGUF, and vLLM 8c17e469 with publisher weights b85c3cfa. GPU driver 595.99.02, 200 W power limit, one sequence, context 2,048, FP16 KV, 1,024 identical input tokens and 32 decode steps. Each fresh process runs eight requests per mode; discard the first. LeanSonic and llama.cpp replay the exact token history produced by native greedy vLLM. Each phase includes production of the next-token ID. Greedy runs from the other engines are included in the receipt.

**3 of five mirrored cycles reported.** Each complete cycle contains two fresh processes per engine, in a rotating order followed by its reverse. vLLM prefill and decode use its scheduled-to-first-token and first-to-last-token timers. The other engines use synchronous in-process timers. All exclude model loading. The sampled peak GPU bytes include initialization and vLLM worker descendants.

| Engine / model package | Prefill tok/s | Decode tok/s | Peak GPU bytes | Distributed model bytes |
| --- | ---: | ---: | ---: | ---: |
| LeanSonic / 571 class-split | 8,526.67 | 124.69 | 14,615,052,288 | 11,483,431,150 |
| llama.cpp / IQ3_XXS | 1,879.16 | 78.84 | 13,176,406,016 | 11,663,309,504 |
| vLLM / official Inkling2 | 8,670.59 | 107.68 | 22,756,196,352 | 19,555,785,477 |

The publisher column counts the shard plus configuration and tokenizer assets, excluding the optional 635,801,416-byte MTP file. The main shard alone is 19,526,019,168 bytes. All distributed model assets including MTP total 20,191,586,893 bytes. The GGUF includes its tokenizer. The LeanSonic directory contains the shard and tokenizer. vLLM runs BF16 activations with FP16 KV and a 2,415,919,104-byte explicit KV cache budget. Its engine log confirms the effective KV dtype. The packages share the underlying model but have different quantizations.

## Reproduction

Use the [physical 571-slice model hashes](../receipt.json), the [pinned native GGUF and runtime](https://github.com/tjm8874/LeanSonic/tree/d65d1a6/docs/benchmarks/inkling-native-lowbits), and the vLLM wheel and official checkpoint revision in the [measurement receipt](measurement.json.gz). Use the release build of lean.rs as a leansonic-core example at a1592f3, named `bench`. Compile native.cpp as `llama` against the matching llama.cpp and ggml headers, linking libllama, libggml and libggml-base. Keep the original pinned native runtime libraries in NATIVE_LIB. Set OFFICIAL, LEAN_MODEL, GGUF and VLLM_PYTHON to their respective local paths. The NVML monitor uses the selected GPU UUID in CUDA_VISIBLE_DEVICES.

In a fresh directory containing the two executables, Python files and prompt.ids, enable LEANSONIC_PDL, LEANSONIC_PDL_ALL, LEANSONIC_MXFP8_DEVICE_U4, LEANSONIC_MXFP8_DIRECT_ALL, LEANSONIC_MIXED_EVEN_UP and LEANSONIC_MIXED_EVEN_DOWN. Set LEANSONIC_TERNARY_UP=0, LEANSONIC_TERNARY_DOWN=0, LEANSONIC_KV=fp16, LEANSONIC_SEQS=1, LEANSONIC_CTX=2048 and LEANSONIC_PREFILL_CHUNK=1024. Point LEANSONIC_CUBIN_PATH at the checked SM120 cubin digest in the receipt. Then run:

```sh
tr ',' ' ' <prompt.ids >prompt.words
python3 leg.py anchor vllm
tr ',' ' ' <anchor.ids >anchor.words
mkdir runs
for cycle in 1 2 3 4 5; do
    case "$cycle" in
        1|4) arms="lean llama vllm vllm llama lean" ;;
        2|5) arms="llama vllm lean lean vllm llama" ;;
        3) arms="vllm lean llama llama lean vllm" ;;
    esac
    leg=0
    for arm in $arms; do
        python3 leg.py "runs/$cycle-$leg-$arm" "$arm"
        leg=$((leg+1))
    done
done
```

Check the generated anchor against the supplied anchor.ids and compare all 33 predictions per request. Report prefill as 1,024 divided by the mean elapsed seconds and decode as 32 divided by the mean elapsed seconds, with equal representation per complete process and cycle. The [compressed receipt](measurement.json.gz) records the inputs, model assets, predictions, process peaks, phase timings and executable identities.
