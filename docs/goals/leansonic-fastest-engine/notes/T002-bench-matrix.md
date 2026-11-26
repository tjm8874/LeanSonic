# T002 bench-matrix（Worker, Owner AI: Claude）

- 実装: #36（`leansonic bench-matrix`、`docs/benchmarks/targets.json`）、main にマージ済み
- 計測 commit: `84f67c9`（v0.5.0。#32 NVFP4 int8 MMA、#34 Q8 K16 MMA、#37 split-KV 統合を含む）
- 条件: `LEANSONIC_ATTN_SPLIT=16`（#37 推奨の設定）、`LEANSONIC_CTX=20480`、N=5（1 サンプル = 新しいプロセス 1 つ）、`CUDA_DEVICE_ORDER=PCI_BUS_ID`
- 投機は Basalt と同じ実プロンプト（conv / struct / prose）を使用
- 2026-11-26 19:45〜20:20 UTC。計測中、他のプロセスは GPU を使っていない（`foreign_processes` が空）

## 結果（中央値と 95% CI。全セルでサンプルのばらつきは 1% 未満）

| セル | PRO 5000 | 目標 | 目標比 | PRO 4500 | 目標 | 目標比 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill 2K | 1,447 [1,446–1,448] | 14,000 | 10.3% | 1,080 [1,080–1,083] | 11,500 | 9.4% |
| Prefill 16K | 1,246 [1,243–1,249] | 14,000 | 8.9% | 912 [910–917] | 11,500 | 7.9% |
| Decode 短文（spec off） | 152.4 | 220 | 69% | 129.3 | 149 | 87% |
| Decode 16K（spec off） | 137.4 | 185 | 74% | 117.0 | 125 | 94% |
| 投機 conv / struct / prose 平均 | 210.4 | 310 | 68% | 176.2 | 210 | 84% |
| VRAM（spec off） | 17.26 GiB | ≤16.6 | +0.66 | 17.18 GiB | ≤16.6 | +0.58 |
| 128K / C=64 / KLD | UNMEASURED | | | UNMEASURED | | |

T001（`c0f1e39`）からの変化（PRO 5000）:
- Prefill 2K: 1,001 → 1,447（+45%）
- Decode 16K: 51.6 → 137.4（×2.66、split-KV）
- 投機: 194 → 210

**16GB の 2 枚（5070 Ti / 5060 Ti）は未計測。** cudaserver の 5060 Ti を、Codex の検証プロセスが断続的に使っているためです（PCIe とホストを共有するので、5070 Ti の計測も避けました）。空いた時点で同じコマンドで追加します。

## 補足

- **投機の値は実プロンプトで測る必要があります。** v0.5.0 のレポートには PRO 5000 で DSpark 322.4 tok/s とありますが、Basalt と同じ実プロンプトでは 210.4 tok/s でした。合成プロンプト（`synth:N`）は繰り返しが多く、受理率が高く出ます
- **#37 の split 統合でも、`--spec 0` と `--spec 7` の出力一致を独立に確認しました。** conv / struct / prose / synth:1000 / 1500 / 3000 の 6 種を、G=0 と G=16 の両方で比較し、token ID の md5 が一致しています（PRO 4500）

## ボードの整理（この PR で反映）

- T003（Codex）: #9 への審査コメント（spec との一貫性、SMEM 14,976 B、KLD 比較）で完了
- T004: Gemini が #37 で統合して完了（Owner AI を Gemini に訂正）
- T012（#15）: #37 で、NVRTC builtins の自動読み込み、GEMV の `cols <= 3072` ガード、HTTP の上限とタイムアウトが入ったので完了
- T006（#10）: Codex の #32（NVFP4 int8 MMA）と #34（Q8 K16 MMA）で第 1 段階が進行中。Prefill は目標の約 10% で、未完了のまま
- 次の active: T011（Claude、INT8 KV と 128K）
