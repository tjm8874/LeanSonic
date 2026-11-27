# Lossless scale widths and decode time

The 571-expert model has forty NVFP4 expert scale planes using a fixed six bits per code on the GPU. Encoding groups of 128 consecutive codes with a minimum and a local bit width reduces the aggregate encoded buffers from 955,515,040 to 719,330,832 bytes. One descriptor per expert produces 836,656,032 bytes. All encoders independently decoded every scale value back to the input, including descriptors, alignment padding and terminal read guards in the allocated size. These are scale-buffer sizes. [Every plane, width distribution and byte total](receipt.json).

To test the GPU read cost, we use the same layer-0 mixed gate/up weights for each arm and take scale codes from the real model. A process launches 16 fused gate/up kernels per timed sample, exercising all 128 experts and checking 147,456 Q8 outputs, 9,216 FP32 output scales and 512 gate/up dot products against a double-precision reference. Each experiment has nine alternating four-leg cycles using CUDA events.

| GPU and format | Fixed-six baseline, µs per launch | Candidate, µs per launch | Median within-cycle time ratio |
| --- | ---: | ---: | ---: |
| RTX 5060 Ti, group of 128 scales | 49.246 | 67.553 | 1.372 |
| RTX PRO 4500, one descriptor per expert | 27.083 | 31.721 | 1.172 |

Each pair has its own same-GPU control. The group-of-128 reader is 37% slower in this kernel on the 5060 Ti. Moving the descriptor outside the per-word loads and specializing at five or six bits still costs 17% on the PRO 4500. We retain the fixed-six reader for inference. Model-wide throughput and peak process VRAM for either alternative have not been measured. [Group-of-128 timings](verified.json) · [Expert-width timings](expert/verified.json).

## Reproduction

The full-buffer scan used the [571-expert class-bank package](https://github.com/tjm8874/LeanSonic/tree/4c28635/docs/benchmarks/inkling-embedding-lut/magnitude/44-levels/ranked-experts/class-banks); the kernel fixture used layer 0 from the [earlier 571-expert package](https://github.com/tjm8874/LeanSonic/tree/dd81c6b/docs/benchmarks/inkling-embedding-lut/magnitude/44-levels/ranked-experts). Use engine commit bd048b8 and CUDA 13.0.88. Work in a fresh directory for each variant.

1. Build `scales.rs` and `pack.rs` as examples of leansonic-core. Copy those executables to the working directory as `scales` and `pack`. Run `python3 measure.py "$CLASS_BANK_MODEL"` for the complete scan and `python3 pack.py "$EARLIER_MODEL"` for the fixed weight, scale and amplitude arrays.
2. Build `pack128.rs` with `rustc -O --edition=2021` and run it in that directory. Run `python3 group.py "$ENGINE/kernels/inkling2_sm120.cu"`. Compile the unchanged translation unit as candidate.cubin and grouped.cu as grouped.cubin using `nvcc -I"$ENGINE/kernels" -arch=sm_120a -std=c++17 -O3 --cubin`.
3. Build the measurement driver with `nvcc -std=c++17 -O3 -arch=sm_120a up.cu -lcuda -o up`. Run `./up` on an idle device. Each CSV line reports cycle, leg, format and milliseconds for the 16 launches. For the expert-width experiment use the three corresponding source files in the expert directory and the same model arrays and baseline module.
