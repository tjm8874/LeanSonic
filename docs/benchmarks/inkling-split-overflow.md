# Check long-context split-attention partials without model weights

The fixture launches the production split and combine kernels with uniform attention. Q, K, relative projection, sinks and gate logits are zero. Each attended key has a constant positive or negative value and the sink contributes zero. With N accessible keys, the expected dequantized gated output is `0.5 * value * N / (N + 1)`. The three query rows exercise a partial final query tile. Six key splits exercise the on-disk cubin's split/recombine arithmetic. Both FP16 and INT8 KV use the same values. No LLM weights or text corpus are needed.

Build the cubin at the revision under test and run on a Blackwell GPU with the CUDA driver SDK installed:

```sh
nvcc -std=c++17 -O3 -arch=sm_120a --cubin kernels/inkling2_sm120.cu -o /tmp/inkling-split.cubin
c++ -O2 -std=c++17 tools/check_inkling_split_overflow.cpp -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64/stubs" -lcuda -o /tmp/check-inkling-split
/tmp/check-inkling-split /tmp/inkling-split.cubin
```

The executable returns 0 only if all cases have finite FP16 partials, finite dequantized outputs, and maximum absolute output error below 0.005. Setup or launch errors return 2. Maximum error is computed over finite outputs; the nonfinite counts separately detect infinities and NaNs.

On RTX 5060 Ti, before #110 (`1444305`) both formats passed at 4,096 keys. At 98,304 and 131,072 keys each run produced 55,296 nonfinite partial values and 9,216 nonfinite output values. With `11758a0` all six cases passed. The maximum absolute errors over all heads and query rows were 0.00001400, 0.00004068 and 0.00003050 at the three respective lengths, in both KV formats. The real-model retrieval results are tracked separately in #101.
