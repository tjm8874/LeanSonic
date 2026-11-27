# Direct two-bit gate/up staging in prefill

The first gate/up implementation spent an extra 31.1 ms processing the 1,024-token prompt on RTX 5070 Ti. It expanded a full resident gate/up matrix before every grouped Tensor Core visit. The new module expands only the current weight tile in shared memory. The four-bit and two-bit paths share their MMA, scaling, activation and output logic. The measured 1976fb3 expansion path and its per-request regression are documented in the [full-engine comparison](../decode/).

## Integrated parity

On RTX 5060 Ti and RTX PRO 4500, the old and new modules produced identical 64-token continuations, successor IDs and six full FP32 logit vectors (4,801,392 bytes). Prefixes were 32, 1,024 and 3,072 IDs; the last input cycles the same README fixture three times. The configured workspace was 2,048 tokens on both arms, so the 3,072 prompt spans chunks. The separate direct kernel launches below cover the 160-row specialization. The two GPUs also produced identical FP32 and token fingerprints. [RTX 5060 Ti receipts](5060Ti/) · [RTX PRO 4500 receipts](PRO4500/).

Use 30a41f7 to build this directory's smoke.rs as a release example in leansonic-core. Compile the prior and candidate cubins at 1976fb3 and 30a41f7, respectively, with `nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin kernels/inkling2_sm120.cu` and distinct output names. Follow the [two-bit down parity setup](../../inkling-ternary-down/integrated/) and set LEANSONIC_TERNARY_UP=1, LEANSONIC_TERNARY_DOWN=1, LEANSONIC_CTX=4096 and LEANSONIC_PREFILL_CHUNK=2048. Run each module through the same binary, passing the same model and 1,024-token fixture with unique output prefixes. Compare the .ids and .f32 files byte-for-byte.

## Direct tile coverage

On RTX 5070 Ti and RTX PRO 4500, launch each of the 128- and 160-row tile variants by name using the production module. Eleven expert counts range from 1 to 319, including 63/64/65, 127/128/129 and 159/160/161. The two-bit and nibble arms agree on all 1,585,152 FP32 values per tile shape, with 2,304 guard values preserved per allocation. Inputs use all real layer-0 weight and scale bytes, official expert amplitudes, a fixed activation seed and nontrivial routing. The harness checks finite outputs and substantial nonzero coverage before comparing their exact bits. On RTX PRO 4500, Compute Sanitizer memcheck completed both tile widths with zero errors and reproduced the unsanitized outputs. [Memcheck output](tiles/PRO4500/memcheck.txt) · [Tile checks](tiles/tiles.out) · [GPU and module fingerprints](tiles/).

To reproduce, extract the gate/up plane and amplitudes with the [kernel fixture recipe](../). Build engine.cubin at 30a41f7. In the fixture directory run `nvcc -std=c++17 -O3 tiles.cu -lcuda -o tiles` using the supplied source, then run `./tiles` with the GPU pinned.
