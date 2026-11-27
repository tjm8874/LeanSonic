# Local reconstruction tables for compact MXFP8 experts

The compact shard keeps exactly the same symbols and assignment for every weight. For each set of 16 physical rows, the fitter selects a finite, signed E4M3 reconstruction value for each occupied symbol. It minimizes squared error against the official weights after multiplying by their block scales. The gate and up rows have separate tables. Each zero symbol stays exactly zero. The fit uses only checkpoint weights.

| Files | Actual bytes |
|---|---:|
| model-00001-of-00001.safetensors | 13,623,359,480 |
| mxfp8-group-tables.zip | 59,343 |
| tiktoken/tokenizer.model | 3,386,494 |
| **Distribution total** | **13,626,805,317** |

The reconstruction tables change 361,479,065 out of 4,076,863,488 weights. The same 3,538,944 bytes of lookup tables plus their manifest take 59,343 bytes in a deterministic ZIP. The resulting three-file distribution is 4,972,699 bytes smaller than the original Unsloth Q3_K_XL GGUF. A stock engine reads the **materialized** package, which occupies 13,732,128,366 bytes. Unpack and apply the tables before loading the stock engine; loading the unchanged compact shard alone would bypass the fitted values.


## Reproduce

Use fresh output directories and the compact package from the [parent experiment](README.md). Python 3.11+ and NumPy are required. The distribution command hard-links the two immutable base files; place its output on the same filesystem as COMPACT. Transferring the complete directory includes their full contents. The stock-runtime decoder verifies every reconstructed frame. The measured source shard, table hashes, package bytes, and materialized output digest are in the [receipt](group-tables.json).

```sh
tools=tools/inkling_storage_probe
g=$tools/group_tables
export PYTHONPATH=$PWD/$tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
shard=model-00001-of-00001.safetensors
python3 "$g/fit.py" "$OFFICIAL/$shard" "$COMPACT/$shard" docs/benchmarks/inkling-codebooks/compact-lut.json tables
python3 "$g/package.py" "$COMPACT" tables "$DIST"
python3 -m zipfile -e "$DIST/mxfp8-group-tables.zip" restored-tables
cp "$g/materialize.rs" crates/leansonic-core/examples/mxfp8_group_tables.rs
cargo build --release -p leansonic-core --example mxfp8_group_tables
python3 "$g/apply.py" "$DIST" restored-tables "$PWD/target/release/examples/mxfp8_group_tables" "$STOCK"
LEANSONIC_KV=fp16 LEANSONIC_PDL=1 leansonic eval-kld --weights "$STOCK" --ref "$TEST_KLD" --chunks 100
```
