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
| **Gate 3** | CPU & GPU | `leansonic verify-parity` | **48-visit `0-ULP BATCH IDENTICAL` 検査**（`C = 1` の `SimtReg64` と `C = 64` の `GroupedMmaInt8` が最終出力まで1ビットも違わないこと）。 |
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

# Gate 3: 48-visit Looped MoE における C=1 vs C=64 の 0-ULP 完全ビット一致検証
cargo run --release -p leansonic-cli -- verify-parity

# 第2層チューナ: Blackwell RTX PRO 5000 での離散崖枝刈り＆理論ストリーミング天井試算
cargo run --release -p leansonic-cli -- tune
```

### 2. 第1層 Lean 4 証明の再検証とヘッダ再生成

```bash
cd spec
lake build
lake exe leansonic-gen ..
cd ..
python3 ci/check_lean_axioms.py
```

---

## ロードマップ

- [x] **Phase 0 — 凍結信頼コア（Immutable Trust Core）＆ 5段階CIゲート基盤**: Lean 4による `sm_120` 離散資源則・`0-ULP`・非破壊プール仕様の定義、`.rs`/`.cuh` 自動生成、Gate 0〜4監査スクリプト、GitHub Actionsワークフローの構築。
- [x] **Phase 1 — `Inkling-2-mini` 縦貫通スライス ＆ `Qwen3.8` 共通スケジュール基盤**: `VisitBytecode`（`24 → 48` Looped + `3:1` GDN/QSA）、`HybridSlotPool`（`64` 整列SWAリング + `O(1)` GDN投機ロールバック）、`128B` Planar SoA重み、`0-ULP` 2レジームMoE、`sm_120` CUDA/PTXテンプレート群の投入。
- [ ] **Phase 2 — 実重み `safetensors` ストリーミング・`libcuda` Graph実行・完全無人PRループ稼働**: `Inkling-2-mini` / `Qwen3.8-Flash-Next` 実チェックポイントの `mmap` ローダ、QSAマイクロブロックインデクサ、51B `Engram` ホストDMAリングの接続、およびBlackwell実機ランナーを用いた自律Issue→PR→Merge→Releaseの本稼働。

---

## ライセンス

MIT License — 詳細は [LICENSE](./LICENSE) を参照してください。
