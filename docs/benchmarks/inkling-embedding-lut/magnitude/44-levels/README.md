# A direct size comparison with the native Q4_K input tensor

At a 260,000,000-byte fit target the same magnitude decoder uses 44 representatives and no secondary indices. The final physical frame is 259,021,360 bytes, 1,505,744 bytes smaller than the Q4_K embedding in the native IQ3_XXS GGUF. Across all 463,159,296 official BF16 weights, the relative RMS error is 0.05680667 versus 0.07174278 for Q4_K. The worst absolute error is 0.140625 versus 0.42431641. These figures compare the input tensors.

Against our 452-magnitude fit, this saves 242,527,992 embedding bytes but increases relative RMS error from 0.00273538 to 0.05680667. With the same 217 experts the package is 11,419,458,046 bytes. A separate candidate allocates the recovered space to more experts under the original IQ3_XXS byte budget.

Use the [same recipe](../README.md) with map-260000000.u16 and the packer supporting an empty secondary stream. [Assembled package](receipt.json) · [Independent full readback](check.json) · [Native Q4_K control](../../refit/native-q4).

## Whole-model comparison at the same 217 experts

On one RTX PRO 4500, the 44-level model saves 242,527,992 package bytes over the 452-level model. Each split uses 100 matched WikiText chunks and 25,500 positions against the same BF16 oracle, scorer binary and CUDA module. No experts change between these two packages.

| Split | Magnitudes | Mean KL | Worst KL | BF16 top-1 matches | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 452 | 0.0196107 | 3.1903083 | 23,973 | 7.140181 |
| Test | 44 | 0.0196979 | 2.4135362 | 23,940 | 7.134607 |
| Validation | 452 | 0.0181393 | 1.9770640 | 24,079 | 6.581090 |
| Validation | 44 | 0.0185062 | 3.0118662 | 24,094 | 6.581250 |

The smaller embedding increases mean KL on both splits. Worst KL improves on test and deteriorates on validation. Keep both as separate size/quality candidates while testing the 571-expert allocation at the larger byte budget. [44-level raw scores and executable hashes](corpus.json) · [452-level raw scores](../corpus.json).
