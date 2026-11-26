# LeanSonic

次世代ハイブリッドMoE LLM（**`Inkling-2-mini` → `Inkling-2-Mid`** および **`Qwen3.8-Flash-Next` → `Qwen4`**）を対象とした、**Lean 4形式検証済み・NVIDIA Blackwell専用ミニマムRust推論エンジン ＆ 人間レビュー不要の自律最適化パイプライン**。

**[English](./README.md)** | [日本語ドキュメント](./README.ja.md)

---

## なぜ `LeanSonic` なのか？

従来のLLM推論エンジン開発とAI自律最適化には、2つの本質的な壁がありました。
1. **手書きCUDA最適化における「性能の崖」と潜在バグ**: NVIDIA Blackwell（`sm_120`）では、`64 KiB → 100 KiB` の離散Shared Memoryカーブアウト境界を1バイトでも超えるとL1キャッシュが `64 KiB` から `28 KiB` に激減します。さらに `union Smem` のステージ生存期間エイリアス、レジスタスピル（`STACK > 0`）、48回ループ層で増幅する `1-ULP` のリダクション順序ズレなどが頻繁に発生します。
2. **人間レビューを外した際の「AIによるテスト改ざん（Reward Hacking）」とセキュリティ境界**: 自律エージェントに無制約でPRを作らせると、証明に `sorry` を混ぜたりKLD許容値を緩めてCIを通そうとするほか、公開リポジトリの外部PRやIssue経由でSelf-hosted GPUランナーが乗っ取られるリスクが生じます。

**`LeanSonic`** は、ターゲットを大胆に絞り込んだ**三層ハイブリッド構成**と**凍結信頼コア（Immutable Trust Core）**によって、これらを構造的に解決します。

* **対象GPU**: **NVIDIA Blackwell（`sm_120` / `sm_100`）専用**（旧世代互換の分岐コードを完全排除）。
* **対象LLM系統（次世代本命の2系統を5つの共通プリミティブに統合）**:
  * **`Inkling-2-mini` → `Inkling-2-Mid`**: 物理 `24` 層の重みを `48` 回訪問する **Looped Transformer**（Prelude `6` + Loop Core `12×3` + Coda `6`）、`12` Full Attention + `36` SWA（`window = 1024`）、`128` experts（`top-8`）、`NVFP4` / `MXFP8` / `Q8` Planar SoA、`W = 16` ハイブリッド投機デコード。
  * **`Qwen3.8-Flash-Next` → `Qwen4`**: `3:1` マクロブロック（`3×` **Gated DeltaNet** 線形リカレント層 + `1×` **Qwen Sparse Attention** マイクロブロック選択層）、**4分岐 Gated Residual (GR)**、`512` experts（`top-10`）+ Shared Expert、および **51B `Engram`（N-gram埋め込みテーブル）**のHost RAM非同期プリフェッチ。
* **サードパーティ依存ゼロのミニマムRust構成**: `leansonic-core`・`leansonic-tuner`・`leansonic-cli` はすべてRust標準ライブラリのみで実装されており、ビルドが高速でサプライチェーン攻撃面も最小化されています。

---

## 三層ハイブリッド・アーキテクチャ

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 第1層: Lean 4 形式仕様・不変条件証明・定数コード生成 (`spec/`)               │
│  • `sm_120` 離散資源則: {0,8,16,32,64,100} KiB Carveout と 65,536 レジスタ枠 │
│  • `union Smem` ステージ生存期間非重複（エイリアス根絶）＆ 128B TMA整列証明  │
│  • 全16コード NVFP4 (`E2M1`) -> `2x int8` 無損失 `__byte_perm` LUT 定理      │
│  • 非結合的 IEEE-754 加算のもとでの SIMT <-> `16Bx2` MMA `0-ULP` 完全一致証明│
│  • `lake exe leansonic-gen` により `generated_spec.rs` と `.cuh` を一方向生成│
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 第2層: 離散崖プルーナ ＆ Whole-Model Roofline 探索 (`leansonic-tuner`)       │
│  • L1枯渇カーブアウト、スピル危険構成、広幅クラスタ端数アイドルを事前枝刈り  │
│  • モデル全体のアルゴリズム帯域天井（Prefill / Decode / Speculative）で順位付│
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 第3層: ミニマムRustランタイム ＆ Blackwell PTXテンプレート (`crates/`,`kernels/`)
│  • `VisitBytecode`     : 48-visit Looped と 3:1 GDN/QSA + 4分岐GR を単一実行 │
│  • `HybridSlotPool`    : 64整列KV/SWAリング + `C*(1+W)` O(1) GDN投機ロールバック
│  • `attn_blocklist_tc` : 22-bit (`hi+lo/2048`) FP16 TC Split-KV (Full/SWA/QSA)│
│  • `moe_dual_regime`   : `REG:64` SIMT (<128 pairs) <-> `2xE2M1` MMA (>=128) │
│  • `Async HtoD`        : Spilled Expert と 51B Engram 共通の Late-Wait 転送  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 人間レビューに頼らない「5段階の機械的マージゲート」

公開GitHubリポジトリ上で **`Issue → PR → マージ → リリース`** を人間レビューなしで安全に自律運用するため、リポジトリを**凍結コア（人間オーナーのみ変更可）**と**自律変更領域（AIエージェント変更可）**に物理分離しています。

| ゲート | 実行環境 | 検証スクリプト / ツール | 機械的に強制される合格条件 |
| :--- | :--- | :--- | :--- |
| **Gate 0** | GitHub CPU | `ci/check_immutable_core.py` | `Axioms.lean`・`ci/`・`.github/`・`CODEOWNERS`・`benches/goldens/` への変更を自動PRから完全ブロック（テスト・公理改ざん防止）。 |
| **Gate 1** | GitHub CPU | `lake build` + `ci/check_lean_axioms.py` | 全定理で `sorry`・`sorryAx`・`native_decide`・独自 `axiom` がゼロであることをLeanカーネル環境で検査し、生成 `.rs`/`.cuh` との差分ゼロを強制。 |
| **Gate 2** | CPU / CUDA | `ci/audit_sass.py` + `clippy -D warnings` | `STACK: 0`（スピルゼロ）、ホットループ内 `CALL.REL.NOINC` ゼロ、8-bitスカラ `LDG` ゼロ、Rust `fmt`/`clippy` 警告ゼロ。 |
| **Gate 3** | CPU & GPU | `leansonic verify-parity` | **4大 `0-ULP` 完全ビット一致検査**（① `C=1` vs `C=64` 2レジームMoE、② 72GB常駐 vs 16GB Spilled退避モード、③ Qwen3.8 `Engram` チャンク境界＆投機ロールバック、④ `-p` vs `serve`）。 |
| **Gate 4** | Ephemeral Blackwell | `ci/abba_stat_gate.py` | 同一セッション交互反復（`ABBA × N`）のペアード・ブートストラップ95%信頼区間による統計的非退行・高速化判定（※Forkからの外部PRでは絶対起動しない隔離設計）。 |
| **Gate 5** | Merge Queue | `.github/workflows/autonomous_loop.yml` | 直列Merge Queue通過後の自動マージ、および `v*` タグでの検証済みバイナリ同梱 GitHub Release 自動発行。 |

---

## クイックスタート

### 1. Rustミニマムエンジンのビルドと検証（外部クレート依存ゼロ）

```bash
git clone https://github.com/tjm8874/LeanSonic.git
cd LeanSonic

# 全ワークスペース単体テストの実行
cargo test --workspace

# Lean 4で検証された SM120 / ハイブリッド状態プール定数の確認
cargo run --release -p leansonic-cli -- info

# Inkling-2-Mini (24層->48 visit) と Qwen3.8-Flash-Next (3:1 GDN+QSA+Engram) のスケジュール検証
cargo run --release -p leansonic-cli -- schedules

# Gate 3: 48-visit Looped MoE (C=1 vs C=64) および `-p` vs `serve` の 0-ULP 完全ビット一致検証
cargo run --release -p leansonic-cli -- verify-parity

# 単発プロンプト実行 (`-p` / `--prompt`) — 標準出力ストリームまたは `--json` 構造化出力
cargo run --release -p leansonic-cli -- -p "Explain Looped Transformers on Blackwell" --max-tokens 16
cargo run --release -p leansonic-cli -- -p "Hello Qwen4 hybrid GDN+QSA" --model qwen3.8-flash-next --max-tokens 12 --json

# OpenAI API 互換 HTTP / SSE サーバ起動 (`serve`, 外部クレート依存ゼロ)
cargo run --release -p leansonic-cli -- serve --host 127.0.0.1 --port 8080 --model inkling-2-mini

# 第2層チューナ: Blackwell RTX PRO 5000 での離散崖枝刈り＆理論ストリーミング天井試算
cargo run --release -p leansonic-cli -- tune
```

#### Inkling のテキスト推論用モデルを作る

テキスト推論だけに使う場合は、画像エンコーダーを除いてモデルファイルを小さくできます。言語モデルの重みは元の精度で保存します。変換には Python 3.9 以降と、出力用に約19 GBの空き容量が必要です。

```bash
src=/path/to/official/Inkling-2-Mini
dst=inkling-text
shard=model-00001-of-00001.safetensors
mkdir "$dst" && python3 tools/pack_inkling_text.py "$src/$shard" "$dst/$shard" && cp -R "$src/tiktoken" "$dst/"
```

作成したディレクトリを `--weights` に指定します。DSpark を使う場合は、公式モデルの `mtp.safetensors` も同じディレクトリにコピーしてください。画像入力には元のチェックポイントが必要です。

### 2. OpenAI API 互換エンドポイントの呼び出し（`/v1/models`, `/v1/chat/completions`, SSE）

Lean 4（`ServingAndPrompt.lean`）で帰納法により完全性を証明した SSE 状態機械（`SseProtocolGuard`）と、`2048`-token チャンク Prefill 分離則により、`leansonic -p` と複数スロット同時稼働時の `serve` が **`0-ULP` で完全一致**します。

```bash
# モデル一覧の取得
curl -s http://127.0.0.1:8080/v1/models

# Chat Completions (非ストリーミング — `leansonic -p` と 0-ULP 完全一致)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Hello Blackwell"}],"max_tokens":12}'

# Chat Completions (SSE ストリーミング `text/event-stream` — `finish_reason` + `data: [DONE]` 保証)
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Stream GDN+QSA"}],"max_tokens":12,"stream":true}'
```

### 3. 第1層 Lean 4 証明の再検証とヘッダ再生成

```bash
cd spec
lake build
lake exe leansonic-gen ..
cd ..
python3 ci/check_lean_axioms.py
```

---

## ロードマップ・リリース履歴

- [x] **Phase 0 & 1 (`v0.1.0`) — 凍結信頼コア・5段階CI ＆ `Inkling-2-Mini` 実機GPUエンジン**: Lean 4による `sm_120` 離散資源則・`0-ULP` 証明、全24層・48訪問 Full-GPU `Inkling-2-Mini`（`RTX PRO 4500`: `131.3 tok/s`, `RTX 5070 Ti` Spilled: `65.9 tok/s`）。
- [x] **Cycle 1 (`v0.2.0`) — Multi-Token Batched Prefill**: Planar `Q8_64` `__dp4a` GEMM + Zero-Atomic Grouped MoE（`RTX PRO 4500`: `794.1 tok/s` = `7.14×`, `RTX 5070 Ti`: `337.6 tok/s` = `5.55×`）。
- [x] **Cycle 2 (`v0.3.0`) — 無損失 `DSpark` Full-GPU 投機的デコード**: `mtp.safetensors` GPUドラフタ + `O(1)` Multi-Token Rewind + `0.0000`-ULP ハードウェアFMA完全一致（`RTX PRO 4500`: `326.8 tok/s` = `2.58×`, `RTX 5070 Ti`: `132.0 tok/s` = `2.02×`、凍結コア差分 `0`）。
- [x] **Cycle 3 (`v0.4.0`) — 実重み `106 GB` `Qwen3.8-Flash-Next` / `Qwen4Exp` GPUエンジン**: 22シャード `safetensors` `mmap`、36層 Gated DeltaNet + 12層 QSA Sparse Attention、4分岐 Hyper-Connection、`512`-Expert `Top-10` 4-bit MoE（3階層 VRAM / HostPinned / MmapCompact）、`29.80 GiB` 3-Gram `Engram`/`PLE` 非同期オフロード、48層 CUDA Graph、計 `61` 個の Lean 4 機械証明（凍結コア差分 `0`）。

---

## ライセンス

MIT License — 詳細は [LICENSE](./LICENSE) を参照してください。
