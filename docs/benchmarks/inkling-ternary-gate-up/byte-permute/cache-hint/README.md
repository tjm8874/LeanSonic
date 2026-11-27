# Last-tile L2 eviction hint on packed weights

We tested vector loads with the same cache policy as the raw four-bit path, including evict-first on an expert's final tile, against the two-byte-permute implementation. The model-derived fixtures, output parity checks and data guards were the same as the [byte-permute kernel comparison](../). Five mirrored cycles on RTX 5070 Ti gave 80 paired-shape/tile timing observations. Relative kernel times increased in every case.

| Routing | Tile | Baseline ms | With hint ms | Change |
|---|---:|---:|---:|---:|
| Ragged | 128 | 0.125828 | 0.133139 | +5.81% |
| Ragged | 160 | 0.117716 | 0.119441 | +1.47% |
| Dense | 128 | 1.244612 | 1.373113 | +10.32% |
| Dense | 160 | 1.363233 | 1.448474 | +6.25% |

All four paired intervals for elapsed time lie above zero. Keep the simpler existing load. This experiment changes both the ordinary load cache operator and the last-tile hint, so the regression cannot be attributed to either in isolation. To reproduce, apply generate.py to the byte-permute kernel source, compile both sm_120a modules, and run the existing tiles fixture with the order recorded in result.json.
