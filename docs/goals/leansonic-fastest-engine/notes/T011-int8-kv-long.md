# T011 receipt — INT8 KV と 128K（#14、PR #54）

Owner: Claude。計測日: 2026-11-26。ビルド: main 2399fc5 + PR #54（b093b7a）。#54 は 67156f7 で main にマージ済み。

## 実装
- `LEANSONIC_KV=int8`: Full-attention（global）層の KV を INT8 + 行 × KV ヘッドごとの FP32 スケールにする（opt-in。既定の FP16 経路は main と bit 一致）。
- INT8 専用のカーネル実体を分けた（`ink2_gqa_attn_split_i8`、`ink2_gqa_rel_sink_attn_q8_b_i8`、`ink2_gqa_rel_sink_attn_q8_decode_i8`）。FP16 版の SASS は main と同一。
- `LEANSONIC_ATTN_SPLIT_DECODE`: decode / verify 行だけ split 数を変える（prefill は `LEANSONIC_ATTN_SPLIT`）。
- `bench-matrix --long`（`prefill_128k` / `long_128k_decode`）と、ベンチ専用の `LEANSONIC_BENCH_DECODE_AT`。

objective の「4 段の cp.async.cg と二重バッファの dequant」は採っていない。decode は 1 行ずつの逐次ループが律速で、深さ 1 のレジスタ・パイプライン（INT8 の生ワードとスケールを 1 行先読み）で十分な効果が出たため（PRO 4500 の 32K decode で 9.45 → 8.55 ms/tok）。

## 目標との照合（PRO 5000、128K、実際に 131,072 token を prefill、gen 128）
| セル | 目標 | 結果 | 判定 |
|---|---:|---:|---|
| long_128k_decode | 113 tok/s | 115.8 tok/s（INT8、G=16 / decode 48） | PASS（N=2、ABBA なし） |
| prefill_128k | 9,300 tok/s | 697 tok/s | FAIL（#10 の TC flash attention 待ち） |
| 16GB 機で 128K 起動 | 起動できる | 5070 Ti / 5060 Ti とも FP16 / INT8 で起動 | PASS |

PRO 4500: long_128k_decode 88.4 tok/s（目標 77、PASS）、prefill_128k 492 tok/s（目標 7,600、FAIL）。

## 正確性
- eval-kld（100 chunk × 512）: FP16 mean 0.003151 / max 0.608、INT8 mean 0.003141 / max 0.456、Top-1 97.70% / 97.64%。
- needle（passkey、16K / 32K / 128K × 50% / 90%）: FP16 / INT8 とも 12 / 12 PASS（PRO 4500）。
- `--spec 0` と `--spec 7` の一致: INT8 で G=0 / 16 / 32、G=16 / decode 32、FP16 で G=16 / decode 32。

## 16GB 機（decode、`LEANSONIC_BENCH_DECODE_AT`）
| GPU | 128K FP16 | 128K INT8 | expert 常駐（FP16 → INT8） |
|---|---:|---:|---|
| RTX 5070 Ti | 34.77 ms | 26.08 ms（−25.0%） | 16/24 → 20/24 |
| RTX 5060 Ti | 72.44 ms | 49.60 ms（−31.5%） | 17/24 → 20/24 |

## INT8 prefill が遅い理由（PRO 4500、32K prefill、nsys）
| global 層の attention | 時間 |
|---|---:|
| FP16 split | 約 9.05 s |
| INT8 split_i8（現行） | 10.97 s |
| INT8、スケールの読み込みを外した版（計時のみ） | 10.11 s |
| INT8、int8→f32 を PRMT + FADD にした版 | 11.59 s |

INT8 の遅れのうち約 0.86 s（8%）はスケールの読み込みで、残りは dequant の ALU です。PRMT による変換は I2F より遅くなりました。SIMT の行ループでは改善の余地が小さいので、prefill は TC flash attention（T014）で KV タイルを SMEM に載せる時点で dequant する方針にします。

## 4 GPU の bench-matrix（b093b7a、#8 に掲載）
INT8 KV は 5060 Ti の prefill を 2.1 倍にする（prefill チャンクが 64 → 512 に広がる）。spec_avg の差は DSpark の受理率の差ではない。新しい 12 プロンプトでは受理率が FP16 55.7% / INT8 56.2% で、spec_avg の 3 本は INT8 で生成される文章が変わったことによる振れ。

## 範囲外のファイル
allowed_files 以外に次を変更した。いずれも T011 の検証に必要なもので、凍結コアには触れていない。
- `crates/leansonic-cli/src/bench_matrix.rs`、`docs/benchmarks/targets.json`、`docs/benchmarks/README.md`（T002 で Claude が作成。`--long` セルの追加）
- `kernels/tests/inkling_decode_attention.cpp`（#56 のテスト。カーネルに追加したスケール 2 引数を渡すための修正と `int8` モード）

## 残り
- prefill_128k: #10 Section 18 の `ink2_flash_attn_tc_q8_b` に INT8 KV の読み出し（タイルの SMEM ステージで dequant）を追加する → 新タスク T014。
- 既定の `LEANSONIC_ATTN_SPLIT=0` は decode が遅い（#11 に共有。512 文脈で 27%、128K で約 9 倍）。既定値の変更はオーナー判断。
