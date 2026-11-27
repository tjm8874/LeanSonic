# Parallel byte lookup for ternary tile expansion

On 5070 Ti the [merged end-to-end path](../merged-timing/) still took 1.409 ms longer at 2,048 prompt tokens than full-width gate/up. This experiment changes how eight resident two-bit weights expand to their original E2M1 nibbles. Two register byte permutations look up four low and four high nibbles before merging them. A scalar check enumerates all 65,536 halfwords.

A CUDA-event trial uses real layer-0 weights, scales and amplitudes. The ragged fixture has 1,376 active expert-token pairs across 11 experts. The dense fixture has 16,384 pairs across all 128 experts, with a fixed routing count per expert ranging from 65 to 191. Both use seeded activations. Each timed leg averages 24 kernel launches, after an untimed launch. Five alternating ABBA/BAAB cycles used one 5070 Ti. Each module also launches the full-width kernel and compares every output bit and 2,304 guards at each tile width.

| Routing | Tile rows | Original / lookup ms | Change |
|---|---:|---:|---:|
| Ragged | 128 | 0.126368 / 0.125493 | −0.69% |
| Ragged | 160 | 0.119425 / 0.117423 | −1.68% |
| Dense | 128 | 1.247877 / 1.243252 | −0.37% |
| Dense | 160 | 1.376177 / 1.362994 | −0.96% |

The kernel trial compares 1,585,152 FP32 results on the ragged fixture and 18,874,368 on the dense fixture per arm and tile. The small kernel gains need a full-engine timing before adoption. [Timings and paired intervals](5070Ti.json) · [GPU](gpu.csv) · [Binary hashes](digests)

## Reproduce

Use the [layer-0 fixture recipe](../) for nibbles.bin, ternary.bin, scales.bin and amax.bin. Compile the original 349c7e8 kernels and the changed kernels in this branch with the same nvcc -Ikernels -arch=sm_120a -std=c++17 -O3 --cubin command. Run check.py with Python 3. Build tiles.cu using nvcc -O3 -std=c++17 tiles.cu -lcuda -o tiles. From the fixture directory run each module and routing shape in alternating process order as above.

```sh
./tiles prior.cubin ragged
./tiles trial.cubin ragged
./tiles prior.cubin dense
./tiles trial.cubin dense
```
