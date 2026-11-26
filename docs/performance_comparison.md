# 実機性能比較レポート：LeanSonic vs Basalt vs Octoling（Blackwell 2機種）

本ドキュメントは、**`cuda01`（NVIDIA RTX PRO 4500 Blackwell 32GB）** および **`cuda02`（NVIDIA GeForce RTX 5070 Ti 16GB / Intel Core i5-14400）** の実機2台における `Inkling-2-Mini`（19.5 GB）の3エンジン性能比較および Wave 1 最適化ループ検証結果である。

## 1. `cuda01`：NVIDIA RTX PRO 4500 Blackwell（32 GB VRAM・全24層常駐）

| エンジン / モード | VRAM 使用量 | ロード時間 | 非投機 Decode 短文 (`tok/s`) | 非投機 Decode `@1K` (`tok/s`) | 投機的 Decode (`tok/s`) | Prefill `32 tok` (`tok/s`) | Prefill `1K tok` (`tok/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LeanSonic Wave 0**（初期実装） | 17.74 GiB | 1.89 s | 95.8 (`-p`) / 93.0 (`bench`) | 58.1 | — | 97.8 | 73.2 |
| **LeanSonic Wave 1**（**PR #2 マージ版**） | **16.74 GiB** | **2.22 s** | **131.3** (`-p`) / **128.8** (`bench`) | **91.0** | — | **145.6** (`220ms`) | 116.5 |
| **Octoling** (`mikan-919/octoling`) | 16.58 GiB + Host 4.2 GB | ~2.5 s | 94.7 (`chat`) / 104.0 (`bench`) | 98.5 | — | 329.1 (`chat` `258ms`) | 1,806.7 |
| **Basalt `--spec 0`**（非投機） | 16.60 GiB | 4.0 s | **130.2**（conv `134.6` / struct `129.0` / prose `134.2`） | ~130 | — | — | **8,395** (`16K`: `9,702`) |
| **Basalt Default**（`DSpark` 投機） | 17.00 GiB | 5.0 s | — | — | conv `205.6` / struct `386.1` / prose `167.9` | — | 8,311 (`16K`: `9,624`) |

## 2. `cuda02`：NVIDIA GeForce RTX 5070 Ti（16 GB VRAM・Spilled モード）

| エンジン / モード | VRAM 常駐層数 | ロード時間 | 非投機 Decode 短文 (`tok/s`) | 非投機 Decode 長文 (`tok/s`) | 投機的 Decode (`tok/s`) | Prefill `32 tok` (`tok/s`) | Prefill `1K tok` (`tok/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LeanSonic Wave 0**（初期 Spilled） | 21/24層 (Spilled: 3) | 2.48 s | 43.3 (`-p`) / 42.8 (`bench`) | 33.8 (`@1K`) | — | 43.7 | 38.3 |
| **LeanSonic Wave 1**（**PR #2 Spilled**） | **22/24層 (Spilled: 2)** | **2.86 s** | **65.9** (`-p`) / **65.1** (`bench`) | **54.3** (`@1K`) | — | **69.2** (`462ms`) | 62.1 |
| **Octoling** (`mikan-919/octoling`) | **ビルド不可** | — | **N/A（AVX-512 非対応 CPU）** | **N/A** | **N/A** | **N/A** | **N/A** |
| **Basalt `--spec 0`**（Spilled 非投機） | 動的 Expert Cache | 5.0 s | conv `89.9` / prose `104.5` | struct **`51.5`**（加重平均 **`56.1`**） | — | — | **2,556** (`16K`: `7,018`) |
| **Basalt Default**（Spilled `DSpark`） | 動的 Expert Cache | 6.0 s | — | — | conv `96.7` / struct `80.2` / prose `105.1` | — | 2,291 (`16K`: `6,649`) |

## 3. Gate 4（同一セッション交互実行 `ABBA × 5` 統計検証）および `0-ULP` 一致性

- **`cuda01` (`RTX PRO 4500 32GB`)**: `10.436 ms/tok` → **`7.619 ms/tok`**（レイテンシ **`-26.99%`**, 95% CI `[-27.01%, -26.98%]`, スループット **`+37.0%`**）— **`[PASS] Gate 4`**
- **`cuda02` (`RTX 5070 Ti 16GB`)**: `23.101 ms/tok` → **`15.176 ms/tok`**（レイテンシ **`-34.31%`**, 95% CI `[-34.39%, -34.24%]`, スループット **`+52.2%`**）— **`[PASS] Gate 4`**
- **クロスGPU `0-ULP` 決定論性**: `cuda01`（全24層常駐）と `cuda02`（22層常駐 + 2層 Spilled）で同一プロンプトに対するロジット総和（`1977052.2500` / `2402877.0000`）および生成 128 トークン列が完全一致。
