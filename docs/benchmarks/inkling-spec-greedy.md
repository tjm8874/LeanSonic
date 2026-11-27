# Check DSpark greedy output against regular decoding

On commit `4b238be`, RTX 5070 Ti and RTX 5060 Ti produced identical regular and width-7 speculative continuations for the first 136 tokens of the published conversation prompt. At zero-based index 136 regular decoding produced ID 68154 and DSpark produced ID 68020. The complete 200-token regular sequence matched across cards, as did the complete speculative sequence. [Issue #88](https://github.com/tjm8874/LeanSonic/issues/88) tracks the regression. [Hashes, tested widths and first differences](inkling-spec-greedy.json).

On both GeForce cards, width 3 first differed at index 136 and width 1 at index 181. A separate rebuild of main at `b02cce0` (including PR #89) reproduced the width-7 difference on PRO 5000, 5070 Ti and 5060 Ti, with the same complete regular and speculative output hashes.

Use the official Inkling-2-Mini checkpoint with its `mtp.safetensors` drafter. Download [the conversation prompt from Basalt](https://github.com/jesdga95/basalt/blob/5fe1928d/data/bench/inkling2/conv.ids) as raw bytes. Its SHA-256 is `a77b81cb16f94cdc8baee2073519fd1bf97a85e3533659eeec1c57d81105b373`. On a GPU worker with the release binary and matching cubin:

```bash
export LEANSONIC_KV=fp16 LEANSONIC_CTX=8192 LEANSONIC_CUBIN_PATH="$CUBIN"
python3 tools/check_inkling_spec_greedy.py --engine target/release/leansonic \
    --weights "$MODEL_DIR" --tokens conv.ids --spec 7 --gen 200
```

The command starts a fresh CLI process for each path, checks that the requested DSpark width actually loaded, and compares all token IDs. It prints JSON with the hashes and first differing token. Exit status 0 means identical outputs, 1 means a token difference, and 2 indicates an execution or input error. The 16-token positive control on this prompt passes even on the affected build. Run all 200 tokens when verifying a fix.
