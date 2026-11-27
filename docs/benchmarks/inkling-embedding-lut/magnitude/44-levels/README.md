# A direct size comparison with the native Q4_K input tensor

At a 260,000,000-byte fit target the same magnitude decoder uses 44 representatives and no secondary indices. The final physical frame is 259,021,360 bytes, 1,505,744 bytes smaller than the Q4_K embedding in the native IQ3_XXS GGUF. Across all 463,159,296 official BF16 weights, the relative RMS error is 0.05680667 versus 0.07174278 for Q4_K. The worst absolute error is 0.140625 versus 0.42431641. These figures compare the input tensors; model-level scores remain pending.

Against our 452-magnitude fit, this saves 242,527,992 embedding bytes but increases relative RMS error from 0.00273538 to 0.05680667. With the same 217 experts the package is 11,419,458,046 bytes. A separate candidate allocates the recovered space to more experts under the original IQ3_XXS byte budget.

Use the [same recipe](../README.md) with map-260000000.u16 and the packer supporting an empty secondary stream. [Assembled package](receipt.json) · [Independent full readback](check.json) · [Native Q4_K control](../../refit/native-q4).
