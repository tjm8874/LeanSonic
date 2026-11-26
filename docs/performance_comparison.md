# 実機性能比較レポート：LeanSonic (`v0.1.0`〜`v0.4.0`) vs Basalt vs Octoling

本ドキュメントは、**`cuda01`（NVIDIA RTX PRO 4500 Blackwell 32GB）** および **`cuda02`（NVIDIA GeForce RTX 5070 Ti 16GB / Intel Core i5-14400）** の実機2台における **`Inkling-2-Mini`（19.5 GB + `DSpark` MTP）** および **`Qwen3.8-Flash-Next` / `Qwen4Exp`（106 GB・22シャード + 29.8 GiB `Engram`）** の3エンジン性能比較・自律進化結果である。

## 1. `Inkling-2-Mini`（19.5 GB）：`cuda01` NVIDIA RTX PRO 4500 Blackwell（32 GB VRAM・全24層常駐）

| エンジン / リリース | VRAM 使用量 | ロード時間 | 非投機 Decode (`tok/s`) | 投機的 Decode (`tok/s`) | Prefill `1K tok` (`tok/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **LeanSonic `v0.1.0`**（Wave 1 Solo） | 16.74 GiB | **2.22 s** | **131.3** (`-p`) / **128.8** (`bench`) | — | 116.5 |
| **LeanSonic `v0.2.0`**（Cycle 1 Batched Prefill） | 16.82 GiB | **2.25 s** | **131.3** | — | **794.1**（**`7.14×` 高速化**） |
| **LeanSonic `v0.3.0`**（Cycle 2 `DSpark` `--spec 7`） | 17.12 GiB | **2.41 s** | 126.4 | **325.6 平均 / 326.8 最大**（`-p` 長文 **183.7**, **`2.58×`**） | **794.1** |
| **Octoling** (`mikan-919/octoling`) | 16.58 GiB + Host 4.2 GB | ~2.5 s | 94.7 (`chat`) / 104.0 (`bench`) | —（非対応） | 1,806.7 |
| **Basalt `--spec 0`**（非投機） | 16.60 GiB | 4.0 s | 130.2（conv `134.6` / struct `129.0` / prose `134.2`） | — | **8,395** |
| **Basalt Default**（`DSpark` 投機） | 17.00 GiB | 5.0 s | — | conv `205.6` / struct `386.1` / prose `167.9` | 8,311 |

## 2. `Inkling-2-Mini`（19.5 GB）：`cuda02` NVIDIA GeForce RTX 5070 Ti（16 GB VRAM・Spilled モード）

| エンジン / リリース | VRAM 常駐層数 | ロード時間 | 非投機 Decode (`tok/s`) | 投機的 Decode (`tok/s`) | Prefill `1K tok` (`tok/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **LeanSonic `v0.1.0`**（Wave 1 Spilled） | 22/24層 | **2.86 s** | **65.9** (`-p`) / 54.3 (`@1K`) | — | 62.1 |
| **LeanSonic `v0.2.0`**（Cycle 1 Batched Spilled） | 22/24層 | **2.90 s** | **65.9** | — | **337.6**（**`5.55×` 高速化**） |
| **LeanSonic `v0.3.0`**（Cycle 2 `DSpark` `--spec 7`） | 21/24層 + MTP GPU | **3.05 s** | 64.1 | **129.8 平均 / 132.0 最大**（**`2.02×` 高速化**, **`> Basalt` 全3ワークロード!**） | **337.6** |
| **Octoling** (`mikan-919/octoling`) | **ビルド不可** | — | **N/A（AVX-512 非対応 CPU）** | **N/A** | **N/A** |
| **Basalt `--spec 0`**（Spilled 非投機） | 動的 Expert Cache | 5.0 s | 加重平均 `56.1`（struct `51.5` / conv `89.9`） | — | **2,556** |
| **Basalt Default**（Spilled `DSpark`） | 動的 Expert Cache | 6.0 s | — | 加重平均 `83.9`（conv `96.7` / struct `80.2` / prose `105.1`） | 2,291 |

## 3. `Qwen3.8-Flash-Next` / `Qwen4Exp`（106 GB・48層 GDN+QSA + 512 Experts + 29.8 GiB `Engram`）

| エンジン / 実行構成 | 実行 GPU / リンク | VRAM 常駐状況 | ロード時間 | 非投機 Decode (`tok/s`) | Prefill (`tok/s`) | 備考 |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **LeanSonic `v0.4.0` (HostPinned + Graph)** | **`RTX PRO 4500 32GB`** (`PCIe Gen5 x8`) | **18/48層 VRAM** + 30/48層 Pinned + 29.8GiB Engram | **10.21 s** | **19.1〜20.1**（物理 PCIe Gen5 x8 帯域上限の **69%**） | **20.8** (`32 tok`) | **32 GB 単機で 106 GB モデルを無損失実行** |
| **LeanSonic `v0.4.0` (MmapCompact)** | **`RTX PRO 4500 32GB`** (`PCIe Gen5 x8`) | **18/48層 VRAM** + 30/48層 Zero-Copy `mmap` (`+30MB` stage) | **2.54〜3.14 s** | **12.6〜14.6** | **11.2〜12.4** | 追加ホスト RAM 消費 `~0 GB`、`HostPinned` と `0.0000`-ULP 完全一致 |
| **Octoling** (`mikan-919/octoling`) | — | — | — | **非対応** | **非対応** | `Inkling-2-Mini` 専用 |
| **Basalt `v0.3.3` (`--spec 0` / `--spec 4`)** | `RTX PRO 5000` (**`72GB` VRAM 専有**) | **48/48層 VRAM (`70.3 GiB`)** + Host PLE | — | `134.5` (serial) / `326.9` (`--spec 4`) | `4,715` (`16K`) | 独自変換 `.basalt` (`72,022 MiB` VRAM 使用、32GB 単機では非収容) |

## 4. Gate 4（ABBA 統計ゲート）および完全 `0.0000`-ULP 決定論性の総括
- **Cycle 1 (`v0.2.0`) ABBA**: `1K` Prefill が `cuda01` で **`-86.0%`**（`111.2 → 794.1 tok/s`）、`cuda02` で **`-82.0%`**（`60.8 → 337.6 tok/s`）短縮。
- **Cycle 2 (`v0.3.0`) ABBA**: `32+128 tok` 生成が `cuda01` で **`-61.16%`**（`126.4 → 325.6 tok/s`）、`cuda02` Spilled で **`-50.61%`**（`64.1 → 129.8 tok/s`、Basalt Spilled `80.2〜105.1 tok/s` を超過）短縮。`Solo (B=1)` と `--spec 7` 間は **`0.0000`-ULP 完全ビット一致**（`sum = 1822743.1250`）。
- **Cycle 3 (`v0.4.0`) `0-ULP` 実機確認**: `106 GB` `Qwen3.8-Flash-Next` において、`HostPinned + 48層 CUDA Graph` と `MmapCompact (--no-graph)` の 248,320 語彙ロジット総和が **`-439388.7500 == -439388.7500`（誤差 `0.0000` ULP）** で完全一致。
