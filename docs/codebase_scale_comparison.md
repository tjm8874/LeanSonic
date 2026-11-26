# ソースコード規模・構造比較レポート：LeanSonic vs Basalt vs Octoling

本ドキュメントは、Blackwell世代GPU（`sm_120`）上で `Inkling-2-Mini`（19.5 GB 公式 `safetensors`）を実行する3つの推論エンジン実装――**LeanSonic** (`tjm8874/LeanSonic`)、**Basalt** (`jesdga95/basalt`)、**Octoling** (`mikan-919/octoling`)――のソースコード規模・外部依存・バイナリサイズ・形式検証基盤の定量比較である。

## 1. 全体規模サマリ（計測コミット: LeanSonic `297e420`, Basalt `5b8a82c`, Octoling `f9847e9`）

| 比較項目 | **LeanSonic** | **Basalt** (`jesdga95/basalt`) | **Octoling** (`mikan-919/octoling`) |
| :--- | ---: | ---: | ---: |
| **ソースファイル数**（Docs/Build除く） | **31** | 1,265 | 16 |
| **総ソースコード行数（Total Lines）** | **10,655** | 342,269（**32.1x**） | 7,791（0.73x） |
| **実効コード行数（SLOC）** | **9,298** | 289,343（**31.1x**） | 6,759（0.73x） |
| **`Inkling-2` 実行系コア行数** | **4,563**（自前Tokenizer/FFI込） | 22,079（`src/models/inkling2` のみ） | 7,791（外部82クレート別） |
| **外部パッケージ依存数** | **0 クレート**（純 Rust `std`） | C++ STL + CUDA + Python群 | **82 外部クレート**（`Cargo.lock` 83 pkg） |
| **リリースバイナリサイズ** | **1.2 MB**（+ `204 KB` `.cubin`） | **169 MB** | **25 MB** |
| **Lean 4 形式仕様・機械証明** | **6 ファイル / 1,341 行（30 定理, `0 sorry`）** | なし | なし |
| **自律 CI/CD ゲート** | **Gates 0〜5（754 行）** | CMake + Python テスト群 | なし |
| **CPU ISA 要件** | **任意 CPU（AVX-512 不要）** | x86-64 / ARM64 | **x86-64 AVX-512 必須** |

## 2. 言語別コード行数内訳

| 言語・カテゴリ | **LeanSonic**（Files / Lines / SLOC） | **Basalt**（Files / Lines / SLOC） | **Octoling**（Files / Lines / SLOC） |
| :--- | ---: | ---: | ---: |
| **Rust (`.rs`)** | 15 / **6,903** / 6,124 | 0 / 0 / 0 | 15 / **5,001** / 4,340 |
| **CUDA / PTX (`.cu`, `.cuh`)** | 6 / **1,964** / 1,633 | 172 / **67,991** / 58,788 | 1 / **2,790** / 2,419 |
| **C / C++ (`.cpp`, `.h`)** | 0 / **0** / 0 | 882 / **212,993** / 177,701 | 0 / **0** / 0 |
| **Lean 4 (`.lean`)** | 6 / **1,341** / 1,179 | 0 / 0 / 0 | 0 / 0 / 0 |
| **Python (`.py`)** | 4 / **447** / 362 | 211 / **61,285** / 52,854 | 0 / 0 / 0 |
| **Build / CI 設定** | 8 / **307** / 279 | 12 / **3,247** / 3,049 | 2 / **63** / 58 |

## 3. 要点まとめ
1. **対 Basalt 比 `1 / 32.1` の最小構造**: Lean 4 によるスケジュール・SM120 ハードウェア崖・`0-ULP` 還元木の事前証明により、34万行規模の C++/CUDA コードベースを約1万行（`Inkling-2` 実行核単体では `3,329` 行：`inkling2_gpu.rs` `1,917` 行 + `inkling2_sm120.cu` `1,412` 行）へ凝縮。
2. **完全ゼロ外部依存 (`0` crates / `1.2 MB`)**: `safetensors` `mmap` パーサ、`tiktoken` BPE、`libcuda.so`/`libnvrtc.so` FFI、OpenAI SSE サーバまですべて Rust 標準ライブラリのみで実装し、サプライチェーンリスクとビルド時間を極小化。
3. **100% GPU 実行による高い移植性**: `Octoling` が CPU AVX-512 へ逃がしていた Coda `MXFP8` 層（Layers `20..23`）も `sm_120` GPU 上で直接実行し、AVX-512 非搭載の `cuda02`（Core i5-14400 + RTX 5070 Ti 16GB）でも訪問多重度重み付き Spilled モードでそのまま高速動作する。
