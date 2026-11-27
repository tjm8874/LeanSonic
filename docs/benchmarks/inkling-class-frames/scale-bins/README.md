# Fit the existing mean-scale entropy boundaries to each projection

On the 571-expert split-class package, fitting the existing 12 × 12 tables reduces all 40 expert scale planes from 569,829,792 to 568,120,296 bytes. The published format and reader accept the fitted boundaries. The packer decoded all 1,274,019,840 scale codes and compared each against its source. The assembler read back all 999 final tensor payloads. Actual shard and tokenizer size fell from 11,483,431,150 to **11,481,721,654 bytes**. [Per-plane boundaries and physical trial sizes](receipt.json) · [Final shard and tokenizer hashes](package.json).

Each tile contributes a fixed rotating 1-in-32 sample of its existing left and vertical eight-neighbor features. Three coordinate sweeps fit cuts against sampled conditional symbol entropy. The final decision uses complete encoded frame lengths, including header, entropy tables and padding. Retain the parent on a tie or larger result. An independent experiment encoded upgraded and original experts in separate scale streams on the 36 mixed projections. It saved 285,432 bytes over their original frames. The fitted single-stream frames were smaller on every one of those 36 projections and used 1,231,408 fewer bytes in total than the best per-plane choices in that alternative. We retained the existing decoder.

## Reproduction

Build the offline measurement helper in a disposable checkout of engine 38e8d57. Set RECIPE to the absolute path to this directory, and PARENT to the [split-class model directory](../README.md) containing the shard and tokenizer. The fitting script needs Python 3.11 or later and NumPy. Use a new working directory for each run.

```sh
git checkout --detach 38e8d57
python3 "$RECIPE/prepare_probe.py"
cp "$RECIPE/probe.rs" crates/leansonic-core/examples/scale_refit.rs
cargo build --release -p leansonic-core --example scale_refit
mkdir "$RUN"
cp target/release/examples/scale_refit "$RUN/probe"
cd "$RUN"
python3 "$RECIPE/fit.py" "$PARENT"
python3 "$RECIPE/assemble.py" "$PARENT" . materialized > package.json
```

The assembly receipt includes each final file hash, the actual package byte delta and the independent tensor payload readback count. For the 571 package the shard SHA-256 is `97041b2fa1cfe16504a8529d943f8029ce0864b69f12bead150b1fae6471aded`.
