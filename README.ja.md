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

Inkling はモデルの読み込み時に、入力を何トークンずつ処理するかを決めます。GPU に置く expert の重みを先に決め、作業領域とローカル KV キャッシュに使える空き容量に応じて、64、128、256、512、1,024 トークンから選びます。選択結果は起動ログの `Prefill chunk` に表示されます。作業領域を最小にするには `LEANSONIC_PREFILL_CHUNK=64` を指定してください。他の候補値を指定すると自動選択の上限になります。従来の cubin では 64 トークンで処理します。

`-p` と `bench` では入力の長さに合わせて作業領域を確保します。たとえば65トークンの入力なら上限は128トークンです。HTTPサーバーは起動後にさまざまな長さのリクエストを受けるため、GPUの空き容量に応じて選びます。どのモードでも、`LEANSONIC_PREFILL_CHUNK` を指定すれば、その値が上限になります。

Inkling の KV キャッシュは既定で FP16 です。`LEANSONIC_KV=int8` を指定すると全文脈（global）4層を INT8（行 × KV ヘッドごとに FP32 スケール）で保持し、`LEANSONIC_KV=int8-all` ではスライディングウィンドウのリングも INT8 にします。GPU メモリが 16.5 GiB 以下の環境で、`LEANSONIC_KV` を指定せずに 8,192 トークン（CLI の既定値）を超える文脈長を使うと、自動で `int8` になります。FP16 の global KV が常駐 expert 層を追い出すのを避けるためで、起動ログには `int8 (auto, ...)` と表示されます。FP16 のままにするには `LEANSONIC_KV=fp16` を指定してください。

#### Inkling のテキスト推論用モデルを作る

テキスト推論だけに使う場合は、画像エンコーダーを除いてモデルファイルを小さくできます。言語モデルの重みは元の精度で保存します。変換には Python 3.9 以降と、出力用に約19 GBの空き容量が必要です。

```bash
src=/path/to/official/Inkling-2-Mini
dst=inkling-text
shard=model-00001-of-00001.safetensors
mkdir "$dst" && python3 tools/pack_inkling_text.py "$src/$shard" "$dst/$shard" && cp -R "$src/tiktoken" "$dst/"
```

作成したディレクトリを `--weights` に指定します。DSpark を使う場合は、公式モデルの `mtp.safetensors` も同じディレクトリにコピーしてください。画像入力には元のチェックポイントが必要です。

公式モデルでは、変換コマンドに `--prepack-head` を加えると、さらに436,576,496バイト小さくなります。このオプションはCargoを使い、通常は起動時に計算する出力層のQ8重みとFP32スケールをCPUで事前に保存します。読み込みにはこの形式に対応したLeanSonicが必要です。他のエンジンで使う場合や量子化をやり直す場合に備えて、元のBF16重みも保管してください。

Attentionの176個の重みも事前変換するには、上のコマンドに `--prepack-head --prepack-attention` を指定します。公式モデルのテキスト用ファイルは17,847,272,800バイトになります。出力層をすでに変換したファイルを入力にする場合は、`--prepack-attention` だけで追加変換できます。どちらも変換時にCargoを使います。

配布容量をさらに減らすには、公式ファイルの変換時に `--prepack-head --prepack-attention --prepack-embedding --prepack-mxfp8 --entropy-scales --compact-q8-scales` を指定します。最後のオプションは Attention と出力層の Q8 スケールを辞書圧縮し、読み込み時に元の FP32 値をビット単位で復元します。Inkling-2-Mini のテキスト重みは 16,035,182,496 バイトで、別途 tokenizer が必要です。すでに他のオプションで変換したファイルには `--compact-q8-scales` だけを指定できます。読み込みには辞書形式に対応した LeanSonic を使ってください。

Q8 の整数値も圧縮するには `--compact-q8-codes` を加えます。行列ごとにサイズを比較し、小さくなる場合だけ rANS 形式で保存します。上記のオプションと併用すると重みは 15,978,753,408 バイト、tokenizer を含む配布容量は 15,982,139,902 バイトです。対応するローダーが読み込み時に元の整数値を復元します。RTX 5060 Ti では配布容量が 56,429,088 バイト減り、モデルの読み込みは平均 5.181 秒から 5.785 秒になりました。ファイルが OS のキャッシュにある状態で、毎回プロセスを起動して計 20 回測定しました。[測定結果と再現手順](docs/benchmarks/inkling-rans-q8-codes.json)

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

### 3. ベンチマーク・品質検証ガイド（`leansonic bench` ほか）

`leansonic bench` は、実機 Blackwell GPU 上での Prefill スループット（`tok/s`）、最終ステップの Top-5 ロジット＆チェックサム、および Solo / 投機的デコード速度（`tok/s`・`ms/tok`）を計測する標準コマンドです。`config.json` から **`Inkling-2-Mini`**（`16g` Speed Pack / `32g` Precision Pack）と **`Qwen3.8-Flash-Next`**（`106 GB` 22シャード）を自動判別します。

#### `leansonic bench` コマンド構文・オプション一覧

```bash
leansonic bench \
  --weights <model_dir> \
  --tokens <synth:N | chat:<text> | chat:@prompt.txt | id1,id2,... | @ids.csv> \
  [--gen N] \
  [--repeat R] \
  [--spec 0..7] \
  [--no-graph]
```

| オプション | 既定値 | 説明 |
| :--- | :--- | :--- |
| `-w`, `--weights <dir>` | *(必須)* | `.safetensors` シャードとトークナイザ（Inkling は `tiktoken/`、Qwen3.8 は `tokenizer.json`）を含むディレクトリ。 |
| `-t`, `--tokens <spec>` | *(必須)* | 入力トークン指定（下記4形式に対応）。 |
| `-g`, `--gen <N>` | `32` | Greedy 生成するトークン数（`0` を指定すると Prefill と先頭ロジット検証のみ実行）。 |
| `-r`, `--repeat <R>` | `1` | 同一プロセス内で `reset() -> prefill -> decode` を `R` 回反復（`run 0` は初回確保・ウォームアップ、`run >= 1` がウォーム定常速度）。 |
| `--spec <K>` | `0` | 投機デコード深さ（`0` = Solo Decode、`1..7` = `DFlash 2` / `DSpark` / `Qwen3.8 MTP` による `0-ULP` 検証付き投機デコード）。 |
| `--no-graph` | *(有効)* | CUDA Graph リプレイを無効化（単発デバッグや `nsys` カーネル別プロファイル取得時に使用）。 |

#### `--tokens` の4つの指定形式
1. **`synth:N`**（例: `--tokens synth:2048`）: 決定論的な合成トークン列 `N` 個（`(i * 7 + 100) % 50000`）を生成。外部ファイルなしで Prefill・長文脈スケーリングを再現性100%で計測できます。
2. **`chat:<テキスト>`**（例: `--tokens "chat:Gated DeltaNetの仕組みを3行で説明してください"`）: モデル固有のチャットテンプレート（`effort = none`）を適用してその場でトークナイズします。
3. **`chat:@file.txt`**（例: `--tokens chat:@benches/prompts/code.txt`）: UTF-8 テキストファイルを読み込み、1ターンのユーザー入力としてトークナイズします。
4. **`id1,id2,...` または `@ids.csv`**（例: `--tokens 100,107,114` / `--tokens @fixture.ids`）: カンマ区切りのトークン ID 列を直接 `prefill_tokens` に渡します。

#### 実践的な計測レシピ集

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 1. Inkling-2-Mini (32g 精度追求パック / 16g 高速パック): 2K Prefill + Solo vs DFlash 2 投機デコード比較
leansonic bench --weights /path/to/inkling-32g --tokens synth:2048 --gen 128 --repeat 3
leansonic bench --weights /path/to/inkling-32g --tokens "chat:Rustで非再帰クイックソートを実装してください" --spec 7 --gen 128 --repeat 3

# 2. Qwen3.8-Flash-Next (106 GB Hybrid GDN + QSA + 512-Expert MoE + 30 GiB Engram): Batched Prefill & Solo Decode
leansonic bench --weights /path/to/qwen3.8-flash-next --tokens synth:512 --gen 64 --repeat 3
leansonic bench --weights /path/to/qwen3.8-flash-next --tokens synth:2048 --gen 64 --repeat 3

# 3. 32K 長文脈位置での Decode レイテンシ直接計測（32K Prefill を待たずに位置のみジャンプ）
LEANSONIC_CTX=36864 LEANSONIC_BENCH_DECODE_AT=32768 \
  leansonic bench --weights /path/to/qwen3.8-flash-next --tokens synth:64 --gen 64 --repeat 3
```

#### 主な環境変数（`bench` / `-p` / `serve` 共通）

| 環境変数 | 対象 | 用途 |
| :--- | :--- | :--- |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` | 全GPU実行 | マルチGPU環境で `CUDA_VISIBLE_DEVICES` を `nvidia-smi` の物理 PCIe 番号と一致させる。 |
| `LEANSONIC_CTX=<tokens>` | Inkling / Qwen3.8 | 最大 KV・QSA 文脈長を指定（Qwen3.8 は必要に応じて上限まで自動拡張）。 |
| `LEANSONIC_BENCH_DECODE_AT=<pos>` | `bench` | Prefill 後にデコード位置を `<pos>` へ移動し、長文脈時の Decode 速度のみを高速測定。 |
| `LEANSONIC_EVAL_DUMP=<file.f32>` | `bench` | 全語彙 FP32 ロジットをバイナリ出力（`LEANSONIC_EVAL_FORCE=id1,id2,...` と併用し KLD / `0-ULP` 検証に使用）。 |
| `LEANSONIC_PREFILL_CHUNK=<64..1024>` | Inkling | Batched Prefill のチャンク幅を固定または上限設定。 |
| `LEANSONIC_KV=<fp16\|int8\|int8-all>` | Inkling | KV キャッシュ精度切替（`<= 16.5 GiB` GPU で `> 8K` 文脈時は `int8` が自動選択）。 |
| `LEANSONIC_DFLASH2_DIR=<dir>` | Inkling `--spec` | 5層 `DFlash 2` ブロック拡散ドラフタのディレクトリを明示指定。 |
| `LEANSONIC_CUBIN_PATH` / `LEANSONIC_QWEN38_CUBIN_PATH` | Inkling / Qwen3.8 | 実行バイナリ横の `.cubin` 自動探索を上書き指定。 |

#### 関連ベンチマーク・品質検証サブコマンド（`bench-matrix`, `multiseq`, `canary`, `eval-kld`）

```bash
# 全GPU一括マトリクス計測（中央値＋ブートストラップ95%信頼区間を targets.json と自動照合）
leansonic bench-matrix --weights "$MODEL_DIR" --gpus all --n 5 --out host_results.json

# 複数スロット連続バッチ実行＆ C=1 == C=N 状態非干渉チェック（Inkling-2-Mini）
leansonic multiseq --weights "$INKLING_DIR" --tokens synth:256 --seqs 16 --gen 64

# 定数プロンプト ("Tokyo") + 16K Needle検索 ("RAVEN-731") + --spec 0 == --spec 7 完全一致ゲート
leansonic canary --weights "$MODEL_DIR"
leansonic canary --weights "$INKLING_DIR" --long   # 128K ("MARSH-527" / "CORAL-841") 長文脈検索も実行

# BF16 参照ロジットに対する 100チャンク KL情報量（KLD）・Perplexity 品質監査
leansonic eval-kld --weights "$MODEL_DIR" --ref "$REF_KLD"
```

### 4. 第1層 Lean 4 証明の再検証とヘッダ再生成

```bash
cd spec
lake build
lake exe leansonic-gen ..
cd ..
python3 ci/check_lean_axioms.py
```

---

## コントリビューション ＆ AIエージェント用スキル

人間開発者・自律AIエージェント（Antigravity / Claude Code / Codex）ともに、共通の5段階ゲートと凍結信頼コア規則に従って開発を行います。
- **[CONTRIBUTING.md](./CONTRIBUTING.md)**: Gate 0〜4 のローカル検証手順、`sm_120` 離散資源則、必須の Exact-Size `compute-sanitizer --padding 65536` ＋ Planted-Fault Red Control、共有GPUの `flock` 排他ロック規約、公開リポジトリのプライバシー保護規則を記載しています。
- **[`.agents/skills/leansonic_contribute/SKILL.md`](./.agents/skills/leansonic_contribute/SKILL.md)** (`leansonic_contribute`): AIコーディングエージェントが自動で読み込めるワークスペーススキルです。

---

## 版ごとのリリースレポート（`docs/releases/` — 英語・日本語）

各バージョンの設計・最適化手法・実機検証レポートを **英語・日本語の両方** で [`docs/releases/`](./docs/releases/) に収録しています。

- **`v0.1.0` – `v0.4.0` 総括**: [日本語](./docs/releases/v0.1.0_to_v0.4.0_retrospective.ja.md) | [English](./docs/releases/v0.1.0_to_v0.4.0_retrospective.md) — 凍結信頼コア・Batched Prefill・`DSpark` 投機デコード・初代 `106 GB` `Qwen3.8-Flash-Next` GPUエンジン
- **`v0.5.0`**: [日本語](./docs/releases/v0.5.0_report.ja.md) | [English](./docs/releases/v0.5.0_report.md) — `16 GB` Spilled VRAM 動的配分 ＆ 無損失 `Inkling-2-Mini` シャード圧縮
- **`v0.6.0`**: [日本語](./docs/releases/v0.6.0_report.ja.md) | [English](./docs/releases/v0.6.0_report.md) — マルチスロット連続バッチ（`C = 1..64`）＆ `128K` 長文脈 Split-KV
- **`v0.7.0`**: [日本語](./docs/releases/v0.7.0_report.ja.md) | [English](./docs/releases/v0.7.0_report.md) — `INT8` Global KV キャッシュ ＆ 広幅 `B = 1024` Grouped MoE Prefill
- **`v0.8.0`**: [日本語](./docs/releases/v0.8.0_report.ja.md) | [English](./docs/releases/v0.8.0_report.md) — `DFlash 2` ブロック拡散投機デコード ＆ Exact-Size Sanitizer ゲート
- **`v0.9.0`**: [日本語](./docs/releases/v0.9.0_report.ja.md) | [English](./docs/releases/v0.9.0_report.md) — `PDL` オーバーラップ・GPU圧縮スケール常駐 ＆ Blackwell 4カード統合検証
- **`v0.10.0`**: [日本語](./docs/releases/v0.10.0_report.ja.md) | [English](./docs/releases/v0.10.0_report.md) — `16 GB` Spilled 機向け適応型 `spec_k` ガバナ ＆ `DFlash 2` 標準化
- **`v0.11.0`（`Inkling-2-Mini` 完成版）**: [日本語](./docs/releases/v0.11.0_report.ja.md) | [English](./docs/releases/v0.11.0_report.md) — `16g` 高速パック（`10,222 tok/s` Prefill / `235.9 tok/s` Spec・Spill `0`）＆ `32g` 精度追求パック（`15,057 tok/s` Prefill / `313.9 tok/s` Spec / `mean_kld = 0.002989`）二階層完成
- **`v1.0.0`（正式版 GA — `Qwen3.8-Flash-Next` & `Inkling-2-Mini` 統合完成）**: [日本語](./docs/releases/v1.0.0_report.ja.md) | [English](./docs/releases/v1.0.0_report.md) — `106 GB` `Qwen3.8-Flash-Next` 完全対応（`48/48` `Z24` VRAM 常駐・`7.9 秒` 起動・`504.4 tok/s` Prefill・`179.6〜200.8 tok/s` VRAM 常駐 `MTP` `--spec 1`・`PRO 4500` `CalibratedMixed` `68.7 tok/s`・Lean 4 `212` 定理）

---

## ライセンス

MIT License — 詳細は [LICENSE](./LICENSE) を参照してください。
