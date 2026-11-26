# LeanSonic：4 GPU で外部エンジン以上の速度・最小 VRAM・証明された正しさを持つ Inkling-2-Mini 推論エンジン

## Objective

LeanSonic（main = `f34a08e` 以降）を、Inkling-2-Mini で次の 3 点を同時に満たす推論エンジンに仕上げる。

- **世界最速**: 同じ GPU 上で、外部エンジンの実測最良値以上
- **最軽量・メモリ節約**: 依存ゼロで、VRAM を最小にする
- **高信頼性**: 0-ULP ゲートと KLD ゲートで正しさを証明する

対象は RTX PRO 5000 72GB から、eGPU 相当（PCIe Gen4 x4）の RTX 5060 Ti 16GB までの 4 枚で、同じリリースバイナリで動かす。Gemini・Codex・Claude の 3 AI で協働する。

## Original Request

> Goal Prep で、世界最速・最軽量・高信頼性・メモリ節約、RTX Pro でも eGPU 相当の 5060 Ti でも動く LLM 推論エンジン構築（3 AI 協働）

ユーザーが確定させた詳細（目標表・制約・役割分担）は、Intake Summary とこの文書の各節に保存してある。

## Intake Summary

- Input shape: `existing_plan`（Issue #5・#6・#8〜#15、PR #9 が既にある）
- Audience: リポジトリのオーナー（Toyo）と、実装に参加する 3 AI（Gemini・Codex・Claude）
- Authority: `approved`。tjm8874/LeanSonic での Issue・PR・コメント作成は承認済み。main へのマージは Gate 0〜5 を通った PR だけ
- Proof type: `metric`（＋ KLD・needle の `test`）
- Completion proof: 目標表の全セルについて、ABBA N≥5 の JSON（commit・GPU UUID・driver・電力上限・入力 token ID 込み）と、KLD / needle のレポートが `docs/benchmarks/` にあり、すべての PR がマージ済みで、Release v0.3.0 にバイナリが添付されていること
- Goal oracle: `leansonic bench-matrix --gpus all` の出力が、4 GPU × 目標表の全セルで PASS になること（未実装なので T002 で作る）
- Likely misfire:
  1. 1 枚の GPU や合成プロンプトだけで「最速」と判断する
  2. 精度を落とした量子化や近似で速度を稼ぐ
  3. GPU 番号の取り違えで、別の SKU の結果を記録する（CUDA の既定 FASTEST_FIRST）
  4. 他の AI の計測と同じ GPU を使い、結果を汚す
  5. 計画や Issue の作成だけで完了扱いにする
- Blind spots considered:
  - 5060 Ti の目標は外部の実測が無い推定値。最初に計測して補正する
  - 16GB 機では Spilled の転送（Gen4 x16 / x4）が支配的
  - 48 visit のループで、1-ULP の差が logits に最大 ±4 ほど増幅される
  - cudaserver には既存の Python サービスがある
  - Gemini が #5 を別ブランチで作業中
  - 凍結コアは変更できない
- Existing plan facts:
  - 優先順: P0 #9 → #10、P1 #11 → #5 → #12 → #13、P2 #14、P3 #15
  - 律速（PRO 5000、Claude 実測）: Prefill は grouped MoE の dp4a SIMT が約 56%（Tensor Core 未使用、ピークの約 2%）。Decode は 1 トークンあたり約 1,157 kernel、うち glue が約 2.3 ms（33%）
  - 現状（f34a08e ＋ PR #9、PRO 5000）: Prefill 1K 1,036 tok/s、Decode 151 tok/s、16K 136 tok/s。GPU 版 serve は無く、投機は未実装

## 目標表（Issue #8。Inkling-2-Mini、FP16 KV、単一リクエスト、tok/s）

| 指標 | PRO 5000 72GB | PRO 4500 32GB | 5070 Ti 16GB（Spilled） | 5060 Ti 16GB（Gen4 x4、Spilled） |
|---|---|---|---|---|
| Prefill 2K〜16K | ≥14,000 | ≥11,500 | ≥3,700（2K）/ 5,300（16K） | ≥1,800 |
| 非投機 Decode 短文 | ≥220 | ≥149 | ≥105 | ≥45 |
| 非投機 Decode 16K | ≥185 | ≥125 | ≥85 | ≥40 |
| 投機 DFlash2 / DSpark（conv / struct / prose の平均） | ≥310 | ≥210 | ≥100 | 計測して記録 |
| 長文 128K の Decode / Prefill | ≥113 / ≥9,300 | 帯域比で換算 | KV INT8 で起動できる | KV INT8 で起動できる |
| 同時並列 C=64 の合計 | ≥1,340 | ≥910 | 計測して記録 | 計測して記録 |
| VRAM（CTX 20K、非投機） | ≤16.6 GiB | ≤16.6 GiB | 常駐層を最大化（現状 22/24） | 常駐層を最大化（現状 22/24） |
| 正確性（BF16 参照の 100-chunk） | KLD 平均 ≤0.003・最大 ≤0.05・Top-1 ≥97.9%、needle 50% / 90% 位置で完全一致 | 同左 | 同左 | 同左 |

判定: N≥5 の ABBA の中央値で、95% ブートストラップ CI の下限が目標以上なら PASS。「計測して記録」のセルは、値が記録されていれば PASS。

## Goal Oracle

The oracle for this goal is:

`leansonic bench-matrix --gpus all` が 4 GPU × 目標表の全セルで PASS を出し、その JSON と KLD / needle レポートが docs/benchmarks/ にコミットされていること

The PM must keep comparing task receipts to this oracle. Planning, discovery, a passing tiny slice, or a clean-looking board is not enough. The goal finishes only when a final Judge/PM audit maps receipts and verification back to this oracle and records `full_outcome_complete: true`.

## Goal Kind

`existing_plan`

## Current Tranche

継続実行する。各 tranche の終わりに bench-matrix を実行し、FAIL の多い分野の順に次の Worker パッケージを選ぶ。最初の tranche は「計測基盤（T001 ベースライン・T002 bench-matrix・T005 正確性ハーネス）と P0（#9 → #10）」。

## 3 AI の役割（assignee の割り当て方針）

GoalBuddy の assignee（Scout / Judge / Worker / PM）は役割の種類を表す。実際にどの AI が担当するかは、各タスクの constraints の `Owner AI:` で指定する。

| AI | 主な役割 | 担当 |
|---|---|---|
| Gemini | 主実装（Worker） | エンジン本体の kernel・Rust・Lean。#5（投機）を継続し、#10・#11・#12 の実装 |
| Codex | 検証（Judge）＋ CUDA 支援（Worker） | #6 の窓口。計測条件のレビュー、compute-sanitizer・SASS 監査、rollback / 境界テスト、正確性ハーネス（#13）、堅牢性（#15） |
| Claude | 調査（Scout）＋ 判定（Judge）＋ 限定的な Worker | 外部比較と目標の更新（#8）、4 GPU での ABBA 計測、nsys による律速の特定、bench-matrix（T002）、split-KV（#9）、長文（#14）の試作 |

並列作業: ユーザーが 3 AI での協働を明示的に依頼しているので、**allowed_files が重ならない Worker 同士**に限り、AI ごとに 1 本ずつ（最大 3 本）同時に進めてよい。state.yaml の `active_task` は PM が集中して見ているタスクを指し、他の AI の進行中タスクは `queued` のまま `constraints` に「進行中（branch 名）」と書く。

## Non-Negotiable Constraints

- 凍結コア（`spec/LeanSonic/Axioms.lean`、`ci/`、`.github/`、`CODEOWNERS`、`benches/goldens/`）は変更しない。Gate 0〜5 を通す
- Lean に `sorry` / `native_decide` / 独自 axiom を入れない。新しい kernel には `ValidSm120Kernel` の証明を付ける
- Rust はサードパーティ依存ゼロ。`clippy -D warnings`、`fmt`、SASS で `STACK: 0`、ホットループの `CALL.REL.NOINC` はゼロ
- 構造だけの変更は 0-ULP（出力ハッシュを固定）。数値が変わる変更は 100-chunk KLD ゲートで判定し、PR に区分を明記する
- 計測では `CUDA_VISIBLE_DEVICES=GPU-<UUID>` か `CUDA_DEVICE_ORDER=PCI_BUS_ID` を必ず指定する（cuda01 では既定の順序が nvidia-smi と逆）。計測前に `nvidia-smi --query-compute-apps` で使用状況を確認し、他の計測と GPU を共有しない。GPU ごとに `flock` でロックする
- cudaserver の既存 Python サービス 2 つは止めない
- 公開 Issue / PR に内部 IP や GPU UUID の実値を書かない
- 1 回の実験で変えるのは 1 要素だけ。却下した試行も notes に記録する
- Decode が退行する Prefill 改善は採用しない（Decode Comes First）
- 精度を落とす量子化や近似で目標を達成しない
- 4 枚の GPU のメモリを束ねた単一モデル実行は扱わない（P2P 帯域を実測するまで）

## Non-Goals

- Inkling-2-Mini 以外のモデルの最適化（Qwen 系は共通基盤を壊さない範囲だけ）
- Blackwell（sm_120 / sm_100）以外の GPU 対応

## 使うスキル・ツール

- cuda-optimization スキル（Basalt の実績表、`cuda_audit.py` の `model-ceiling` / `wave-audit` / `abba` / `sass-audit`、`run_cuda_selftest.sh`）
- nsys / ncu / compute-sanitizer、gh（REST API）、Claude 用の GPU SSH Runner
- 外部エンジンの既存成果物（cuda01 の `/models` 以下、読み取り専用）: Basalt Wave 4 の ABBA、vLLM / SGLang / TokenSpeed、llama.cpp、DFlash2 drafter

## Stop Rule

Stop only when a final audit proves the full original outcome is complete.

Do not stop after planning, discovery, or Judge selection if a safe Worker task can be activated.

Do not stop after a single verified Worker package when the broader owner outcome still has safe local follow-up work. Advance the board to the next highest-leverage safe Worker package and continue unless a phase, risk, rejected-verification, ambiguity, or final-completion review is due.

次の場合はエスカレーションする（ユーザーに確認する）:
- 凍結コアやゲートの閾値の変更が必要になったとき
- 精度目標と速度目標が両立しないと Judge が判断したとき
- 同じ検証が 2 回続けて失敗したとき
- GPU やサービスの停止が必要になったとき

これらは該当するタスクだけを blocked にし、他の安全な作業は続ける。

## Slice Sizing

Safe means bounded, explicit, verified, and reversible. It does not mean tiny. A good task is the largest safe useful slice. 1 つの kernel 群（例: grouped MoE の int8 MMA 化）を 1 パッケージにし、ファイル単位に細切れにしない。

## Board Health

```bash
node <skill-path>/scripts/check-goal-state.mjs docs/goals/leansonic-fastest-engine
```

## Canonical Board

Machine truth lives at:

`docs/goals/leansonic-fastest-engine/state.yaml`

If this charter and `state.yaml` disagree, `state.yaml` wins for task status, active task, receipts, verification freshness, and completion truth.

## Run Command

```text
Codex: /goal Follow docs/goals/leansonic-fastest-engine/goal.md.
Claude Code: /goalbuddy Follow docs/goals/leansonic-fastest-engine/goal.md.
```

## PM Loop

On every `/goal` continuation:

1. Read this charter, and follow the GoalBuddy execution contract when available.
2. Read `state.yaml`.
3. Run the bundled GoalBuddy update checker when available and mention a newer version without blocking.
4. Re-check the intake: original request, input shape, authority, proof, blind spots, existing plan facts, and likely misfire.
5. Work only on the active board task.
6. Assign Scout, Judge, Worker, or PM according to the task, and respect the `Owner AI:` line.
7. Write a compact task receipt.
8. Update the board.
9. If safe local work remains, choose the next largest reversible Worker package and continue unless blocked.
10. Record every Issue / PR in a task receipt (publishing is pre-approved for tjm8874/LeanSonic).
11. Review at phase, risk, rejected-verification, ambiguity, or final-completion boundaries.
12. Before ending the host turn, run `node <skill-path>/scripts/check-can-stop.mjs docs/goals/leansonic-fastest-engine`.

Issue and PR handoffs are supporting artifacts. `state.yaml` remains authoritative.
