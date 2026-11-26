# T001 ベースライン計測（Scout, Owner AI: Claude）

- 計測日時: 2026-11-26 18:25〜18:45 UTC
- commit: `c0f1e39`（v0.4.0。#19 DSpark と #28 Qwen3.8 の後、#26・#27・#29 の前）
- binary: `cargo build --release` を strip したもの。sha256 は `ddf35ac8ed1d…`
- 条件:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID`。エンジンのログに出るデバイス名と照合済み
  - `LEANSONIC_CTX=20480`、GPU ごとに `flock`
  - `bench --repeat 5`（1 プロセス内で 5 回）。この T001 は 1 構成だけの計測なので ABBA ではない
- 非投機は `--spec 0` を明示した。main は `mtp.safetensors` があると既定で `--spec 7` になる
- 投機は Basalt の DFlash2 / DSpark 比較（N=5 ABBA）と同じ実プロンプト（`data/bench/inkling2/{conv,struct,prose}.ids`、38 / 112 / 21 tokens）で、`--spec 7 --gen 256`
- 生ログ: 各ホストの `~/leansonic-claude-20261126/t001/`（`P5000.log`・`P4500.log`・`R5070Ti.log`・`nsys/`）

## 1. 現状と目標（中央値 N=5、各セルのばらつきは max−min で 1% 以内。※ は 6〜9%）

| 指標 | PRO 5000 | 目標比 | PRO 4500 | 目標比 | 5070 Ti（21/24 常駐） | 目標比 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill 2K（tok/s） | 1,001 | 7.2%（×14.0） | 779 | 6.8% | 259 | 7.0% |
| Prefill 16K | 853 | 6.1%（×16.4） | 630 | 5.5% | 230 ※ | 4.3% |
| 非投機 Decode 短文（depth 32） | 149.0（6.71 ms） | 68% | 128.0（7.81 ms） | 86% | 50.9（19.65 ms） | 48% |
| 非投機 Decode depth 2K | 91.4（10.95 ms） | — | 86.4（11.57 ms） | — | 43.2 | — |
| 非投機 Decode 16K | 51.6（19.39 ms） | 28%（×3.6） | 52.3（19.11 ms） | 42% | 32.1 ※ | 38% |
| 投機 DSpark K=7：conv / struct / prose | 181.9 / 241.2 / 159.3 | 平均 194.1 = 63% | 154.6 / 203.7 / 136.5 | 平均 164.9 = 79% | 57.5 / 71.8 / 53.2 ※ | 平均 60.8 = 61% |
| VRAM（spec off / on） | 17.20 / 17.66 GiB | +0.6 GiB | 17.12 / 17.58 GiB | +0.5 GiB | 14.69 / 15.15 GiB | — |

※ 5070 Ti の 16K と投機の計測中に、同じホストの 5060 Ti で Codex の `place-actual` 検証が始まりました。PCIe とホスト側の競合による汚れの可能性があります（短文と 2K はその前の計測で、ばらつき 0%）。

**5060 Ti は今回計測できませんでした。** Codex の検証が VRAM 15.7 GiB を使用中で、`cuCtxCreate` が CUDA error 2 で失敗しました（Codex 側の処理には影響していません）。参考値は f34a08e、CTX 8192、22/24 常駐で、Decode 32.3 tok/s（30.96 ms）、Prefill 1K 147 tok/s です。

## 2. 分かったこと

1. **Prefill が最大の差（×14〜18）**: nsys（PRO 5000、2K）では、grouped MoE が NVFP4 の up+down で 43.1%、MXFP8 で 14.8%、合計 58% を占めます。続いて `q8_dp4a_gemm_b` 12.8%、`gqa_rel_sink_attn_b` 10.9%、ctrl 8.1%。PRO 4500 も同じ構成です → T006（#10）が最優先
2. **非投機 Decode は文脈の長さで急に遅くなる**: PRO 5000 で 6.71（depth 32）→ 10.95（2K）→ 19.39 ms（16K）。原因は 24 CTA の逐次 attention で、PR #9（split-KV）を入れると 16K が 7.35 ms になります → T003 / T004
3. **投機は「受理率は Basalt 以上、1 step が 1.7〜1.8 倍遅い」**

   | | LeanSonic tok/step | Basalt DSpark tok/window | LeanSonic ms/step | Basalt ms/window |
   |---|---:|---:|---:|---:|
   | conv | 2.53 | 2.10 | 13.9 | 7.62 |
   | struct | 3.77 | 3.61 | 15.6 | 9.03 |
   | prose | 2.06 | 1.83 | 12.9 | 7.39 |

   verify の 1 step（K+1 行）のコストを下げるのが本筋です。具体的には T008 の glue 融合と、split-KV を複数行に拡張すること（T007）
4. **短文 Decode の律速**（PRO 5000、200 トークン、`--cuda-graph-trace=node`）:

   | kernel | 割合 |
   |---|---:|
   | `q8_dp4a_gemv` | 27.8% |
   | MoE NVFP4 の up + down | 24.7% |
   | `gqa_rel_sink_attn`（短い文脈でも） | 9.3% |
   | `depth_mix` | 9.1% |
   | `rmsnorm_quant` / `short_conv4` / `ctrl_down` / `ctrl_apply` / `rmsnorm_d` / `bf16_gemv` | 各 2.7〜3.3% |

   PRO 4500 では `q8_dp4a_gemv` が 32.1% → T008
5. **16GB 機では、確保する文脈長が常駐層の数を決める**: 5070 Ti で f34a08e と c0f1e39 を同じ条件で比べると、どちらも CTX 20480 で 21/24 常駐、20.6 ms/tok でした（以前の 22/24・15.3 ms は CTX 8192 での値）。**コードの退行ではありません。** ただし、文脈の容量を 8K → 20K にしただけで Decode が約 25% 遅くなります。KV を必要な分だけ割り当てる方式か、INT8 KV（T011）が 16GB 機の Decode に直接効きます。#26（実測の空き VRAM で配置する）が main に入ったので、16GB 機は再計測が必要です
6. **VRAM**: spec off で 17.20 GiB。目標の 16.6 GiB（Basalt `--spec 0`）より 0.6 GiB 多い。DSpark ヘッドで +0.46 GiB

## 3. 目標の補正案（5060 Ti）

同じ f34a08e・22/24 常駐での比較:
- Decode: 5070 Ti 15.3 ms 対 5060 Ti 31.0 ms → 比 0.49（帯域比 0.50 とほぼ同じ）
- Prefill 1K: 341 対 147 → 比 0.43（int8 MMA 比 0.475 と PCIe x4 の影響）

そこで、5060 Ti の目標を次のように直すことを提案します。
- Prefill: 2K ≥ 1,600、16K ≥ 2,300（5070 Ti の目標 × 0.43）
- Decode 短文 ≥ 50 / 16K ≥ 40（5070 Ti の目標 × 0.49。16K は据え置き）

## 4. 運用上の発見

- 計測用のロックが AI ごとにばらばらで、GPU の取り合いが起きました（Claude の `/tmp/leansonic-gpu-<host>-<idx>.lock` と、Codex の検証）。**共通のロックパス**を決めて、#6 で合意することを提案します
- main は計測中にも進みました（#26・#27・#29・#22・#31）。T002 の bench-matrix は、commit と binary の sha256 を必ず記録してください

## 5. 次のタスク候補

- T002（bench-matrix）: 上の 6 セル × 4 GPU を自動化し、5060 Ti の計測と 16GB 機の #26 後の再計測を含める
- T003 / T004（PR #9）: Decode 16K を ×2.6 に改善（実測済み）
- T006（#10）: Prefill の 58% を占める grouped MoE の Tensor Core 化
