# ソースコード規模・構造比較レポート：LeanSonic (`v0.1.0`〜`v0.4.0`) vs Basalt vs Octoling

本ドキュメントは、Blackwell世代GPU（`sm_120`）上で **`Inkling-2-Mini`（19.5 GB 公式 `safetensors` + `DSpark` MTP）** および **`Qwen3.8-Flash-Next` / `Qwen4Exp`（106 GB・22シャード + 29.8 GiB `Engram`）** を実行する推論エンジン――**LeanSonic** (`tjm8874/LeanSonic`, `v0.4.0`)、**Basalt** (`jesdga95/basalt`, `5b8a82c`)、**Octoling** (`mikan-919/octoling`, `f9847e9`)――のソースコード規模・外部依存・バイナリサイズ・形式検証基盤の定量比較である。

## 1. 全体規模サマリ（LeanSonic `v0.1.0` → `v0.4.0` vs Basalt vs Octoling）

| 比較項目 | **LeanSonic `v0.1.0`** | **LeanSonic `v0.4.0`**（2モデル+MTP完成版） | **Basalt** (`jesdga95/basalt`) | **Octoling** (`mikan-919/octoling`) |
| :--- | ---: | ---: | ---: | ---: |
| **対応モデル系統** | `Inkling-2-Mini` | **`Inkling-2-Mini` (+`DSpark` MTP) & `Qwen3.8-Flash-Next` (106 GB)** | `Inkling-2`, `Qwen3.8-FN` 等 | `Inkling-2-Mini` のみ |
| **ソースファイル数**（Docs/Build除く） | 31 | **35** | 1,265（**36.1×**） | 16 |
| **総ソースコード行数（Total Lines）** | 10,655 | **20,290** | 342,269（**16.9×**） | 7,791（外部82クレート別） |
| **実効コード行数（SLOC）** | 9,298 | **17,807** | 289,343（**16.2×**） | 6,759 |
| **`Inkling-2` 実行系（Batched+MTP込）** | 4,563 | **8,324**（`inkling2_gpu.rs`+`mtp.rs`+`.cu`） | 22,079（`src/models/inkling2` のみ） | 7,791（非投機のみ） |
| **`Qwen3.8-FN` 実行系（Engram+Spill込）** | —（仕様のみ） | **3,906**（`qwen38_gpu.rs`+`qwen_bpe.rs`+`.cu`） | 数万行（C++/CUDA + Python） | **非対応** |
| **外部パッケージ依存数** | **0 クレート** | **0 クレート**（純 Rust `std` + POSIX `mmap`/`dlopen`） | C++ STL + CUDA + Python群 | **82 外部クレート**（`Cargo.lock` 83 pkg） |
| **リリースバイナリサイズ** | 1.2 MB | **1.8 MB**（+ `588 KB` / `150 KB` `.cubin`） | **169 MB** | **25 MB** |
| **Lean 4 形式仕様・機械証明** | 6 ファイル / 30 定理 | **6 ファイル / 1,683 行（61 定理, `0 sorry`）** | なし | なし |
| **凍結信頼コア（Immutable Trust Core）変更行数** | 初期投入 | **`0` 行変更（`v0.1.0`〜`v0.4.0` 完全不変）** | なし | なし |
| **CPU ISA 要件** | 任意 CPU | **任意 CPU（AVX-512 不要）** | x86-64 / ARM64 | **x86-64 AVX-512 必須** |

## 2. 言語別コード行数内訳（LeanSonic `v0.4.0`）

| 言語・カテゴリ | **LeanSonic `v0.4.0`**（Files / Lines / SLOC） | **Basalt**（Files / Lines / SLOC） | **Octoling**（Files / Lines / SLOC） |
| :--- | ---: | ---: | ---: |
| **Rust (`.rs`)** | 18 / **12,844** / 11,726 | 0 / 0 / 0 | 15 / **5,001** / 4,340 |
| **CUDA / PTX (`.cu`, `.cuh`)** | 7 / **5,316** / 4,579 | 172 / **67,991** / 58,788 | 1 / **2,790** / 2,419 |
| **C / C++ (`.cpp`, `.h`)** | 0 / **0** / 0 | 882 / **212,993** / 177,701 | 0 / **0** / 0 |
| **Lean 4 (`.lean`)** | 6 / **1,683** / 1,140 | 0 / 0 / 0 | 0 / 0 / 0 |
| **Python CI (`.py`)** | 4 / **447** / 362 | 211 / **61,285** / 52,854 | 0 / 0 / 0 |
| **Build / CI 設定** | 11 / **336** / 302 | 12 / **3,247** / 3,049 | 2 / **63** / 58 |

## 3. 要点まとめ
1. **2大次世代アーキテクチャ + 投機的デコードを対 Basalt 比 `1 / 16.9`（`20,290` 行）で完全実装**: `Inkling-2-Mini`（Batched Prefill + `DSpark` Full-GPU MTP 投機デコード）と `106 GB` `Qwen3.8-Flash-Next`（36層 GDN + 12層 QSA + `512` Experts + `29.8 GiB` `Engram`）の両方を、Basalt（`342,269` 行）の `5.9%` のコード量で自己完結実装した。
2. **凍結信頼コア `0` 行変更での 4 連続自律リリース（`v0.1.0` → `v0.4.0`）**: `Axioms.lean`・`ci/`・`.github/`・`CODEOWNERS`・`benches/goldens/` を 1 バイトも変更せぬまま、AI エージェント単独で 61 定理の機械証明拡張と実機 GPU 最適化・リリースを完走した。
3. **外部クレート依存 `0` ＆ 32 GB 単機での `106 GB` モデル稼働**: 多シャード `safetensors` `mmap`、2 種の BPE トークナイザ（`tiktoken` + HuggingFace `tokenizer.json`）、CUDA Graph、3階層 Spilled オフロードまですべて標準ライブラリのみで `1.8 MB` の単一バイナリに凝縮されている。
