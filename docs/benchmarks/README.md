# docs/benchmarks — 目標表と bench-matrix（goal oracle）

`docs/goals/leansonic-fastest-engine` の goal oracle です。4 GPU × 目標表の各セルについて、PASS / FAIL を出します。

## ファイル

| ファイル | 内容 |
|---|---|
| `targets.json` | SKU ごとの目標値（Issue #8・goal.md の目標表と同じ値）。`null` は「計測して記録」のセル |
| `results_*.json` | bench-matrix の結果。**GPU UUID を含むので、公開する前に `uuid` を伏せる** |

## 使い方

```bash
# 1 ホスト分（cuda01 の 2 枚。nvidia-smi の index ＝ PCI_BUS_ID 順）
LEANSONIC_COMMIT=$(git rev-parse --short HEAD) \
leansonic bench-matrix --weights <Inkling-2-Mini dir> --gpus all \
  --spec-prompts <dir with conv.ids / struct.ids / prose.ids> --out results_cuda01.json

# 両ホストの結果を 1 つの表にまとめる（UUID は表示しない）
leansonic bench-matrix --merge results_cuda01.json results_cudaserver.json
```

主なオプション:

| オプション | 既定値 | 説明 |
|---|---|---|
| `--n` | 5 | 1 セルあたりのサンプル数（＝新しいプロセスの数） |
| `--ctx` | 20480 | `LEANSONIC_CTX` |
| `--quick` | — | N=1 にし、16K を省く（動作確認用） |
| `--long` | — | 128K のセル（`prefill_128k` / `long_128k_decode`）も計測する |
| `--c64` | — | `c64_total` を計測する（`multiseq` で 64 系列を共有ラウンドで decode） |
| `--c4-32k <ids>` | — | `c4_32k_code` を計測する（32K トークンの入力を 4 系列で、系列ごとの DSpark draft をまとめて verify。例: `docs/benchmarks/inputs/inkling-code32k.ids`） |
| `--wide-prompts` | — | ディレクトリ内の `*.txt`（`bench --tokens chat:@file` でチェックポイントの tokenizer を通す）と `*.ids` を DSpark で 1 本ずつ計測し、`spec_wide` / `spec_accept_wide` を記録する（例: `docs/benchmarks/prompts`） |
| `--kld` | — | BF16 参照ファイル（`.kld`）を指定すると、GPU ごとに `eval-kld` を 1 回実行して `kld_mean` を埋める（決定的なので N=1）。max / Top-1 / PPL も JSON の `kld` に記録する |
| `--require-idle` | — | 他のプロセスが GPU を使っていたら中止する |
| `--bin` | 自分自身 | 計測する別のバイナリ |
| `--targets` | `docs/benchmarks/targets.json` | 目標値ファイル |

`--spec-prompts` には、Basalt の DFlash2 / DSpark 比較（N=5 ABBA）と同じ実プロンプトを、1 行のカンマ区切り token ID で置きます。cuda01 の `wt-dflash2/data/bench/inkling2/*.ids` の 1 行目です。

## 計測方法

- サンプル 1 つ ＝ 新しいプロセス 1 つ。`bench --repeat 2` の 2 回目（ウォーム）の値を使う
- 子プロセスには `CUDA_DEVICE_ORDER=PCI_BUS_ID` と `CUDA_VISIBLE_DEVICES=<index>` を渡す。nvidia-smi の名前とエンジンが報告した名前が違えば、`errors` に記録する
- GPU ごとに共通ロック `/tmp/leansonic-gpu-<host>-<index>.lock` を `flock` で取得する（#6 で提案）。他のプロセスが GPU を使っていれば `foreign_processes` に記録する
- 判定: 中央値の 95% パーセンタイル・ブートストラップ区間（2,000 回、固定 seed）
  - 大きいほど良い指標は、下限 ≥ 目標で PASS
  - VRAM と KLD は、上限 ≤ 目標で PASS
  - 目標が `null` で値があれば RECORDED、値が無ければ UNMEASURED
- 出力 JSON には、commit（`--commit` か `LEANSONIC_COMMIT`）、バイナリの sha256、driver、電力上限、PCIe、入力 token の FNV-1a、全サンプルを記録する

## セル

| セル | 計測内容 |
|---|---|
| `prefill_2k` / `prefill_16k` | `synth:2048` / `synth:16384`、`--spec 0` |
| `decode_short` | `synth:32`、`--spec 0 --gen 128` |
| `decode_16k` | `synth:16384`、`--spec 0 --gen 128` |
| `spec_avg` | conv / struct / prose、`--spec 7 --gen 256` の平均 |
| `vram_gib` | `decode_short` 実行時の `VRAM used` |
| `prefill_128k` / `long_128k_decode` | `--long` 指定時のみ。`synth:131072`、`--spec 0 --gen 128`、`LEANSONIC_CTX=131584` |
| `spec_wide` / `spec_accept_wide` | `--wide-prompts` 指定時のみ。全プロンプトの `--spec 7 --gen 256` の平均 tok/s と平均受理率（目標なし、RECORDED）。3 本だけの `spec_avg` は生成文の違いで ±10% 程度振れるので、傾向はこちらで見る |
| `kld_mean` | `--kld <ref.kld>` 指定時のみ。`eval-kld`（100 chunk × 512、teacher forcing）の平均 KLD |
| `c64_total` | `--c64` 指定時のみ。`multiseq --seqs 64 --round 64 --gen 64 --no-check`（`synth:400` を末尾 3s トークンずつ削った 64 本、`LEANSONIC_CTX=4096`）。64 行の decode ラウンドを 63 回実行した合計 tok/s。各系列の出力は単独 decode と bit 一致する（#107、`multiseq` の検査モードで確認） |
| `c4_32k_code` | `--c4-32k` 指定時のみ。`multiseq --seqs 4 --round 64 --gen 256 --spec 7 --dspark --no-check`（`LEANSONIC_CTX=33280`、系列 s は入力の末尾 3s トークンを削る）。合計 tok/s。重みディレクトリに `mtp.safetensors` が必要。`inputs/inkling-code32k.ids` は `kernels/inkling2_sm120.cu` と `crates/leansonic-core/src/inkling2_gpu.rs`（`1f70908`）を連結し、Inkling トークナイザで先頭 32,000 トークンにしたもの（#107 の 32K C=4 Code ゲート） |
