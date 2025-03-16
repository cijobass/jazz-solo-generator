# ジャズアドリブジェネレーター

ジャズのコード進行に基づいてアドリブソロを自動生成するStreamlitアプリケーションです。ルールベースの生成アルゴリズムと機械学習ベースの生成の両方を搭載しています。

## 機能

- コード進行の入力（カスタムまたはプリセット）
- ソロラインとベースラインの生成
- 生成されたメロディのMIDIファイル出力
- ブラウザ内での試聴機能
- 楽譜表示機能
- テンポ、オクターブなどのカスタマイズオプション
- 複数のジャズスタイル（スタンダード、ビバップ、モーダル、フュージョン）に対応
- 機械学習を用いた実際のジャズソロの学習と再現（開発中）

## 最新の更新内容

- 大規模MIDIデータセット（10万件以上）からのコード進行抽出機能を追加
- 高速化されたデータ処理パイプラインを実装（処理時間を35倍高速化）
- ルールベース生成と機械学習ベース生成を組み合わせたハイブリッドアプローチを採用

## インストール方法

### 前提条件
- Python 3.10以上
- FluidSynth（オーディオ再生用）
- SoundFontファイル（例：FluidR3_GM_GS.sf2）
- （オプション）LilyPond（高品質な楽譜表示用）
- （機械学習機能用）TensorFlow 2.x または PyTorch

### セットアップ手順

1. リポジトリをクローンする
```bash
git clone https://github.com/yourusername/jazz-solo-generator.git
cd jazz-solo-generator
```

2. 仮想環境を作成して有効化する
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. 必要なライブラリをインストールする
```bash
pip install -r requirements.txt
```

4. FluidSynthとSoundFontファイルのセットアップ
   - Windows: [FluidSynthのリリースページ](https://github.com/FluidSynth/fluidsynth/releases)からダウンロード
   - macOS: `brew install fluid-synth`
   - Linux: `apt-get install fluidsynth`
   
   SoundFontファイル:
   - [FluidR3_GM_GS.sf2](https://musical-artifacts.com/artifacts/733)をダウンロード
   - `C:/Program Files/fluidsynth/soundfonts/`などに配置
   - `src/constants.py`の`SOUNDFONT_PATH`変数を実際のパスに更新

5. LilyPond（オプション - 高品質な楽譜表示用）
   - [LilyPondのウェブサイト](https://lilypond.org/download.html)からダウンロードしてインストール

6. データ処理のための追加ライブラリ（オプション）
   ```bash
   pip install mido tqdm pandas
   ```

## 使用方法

アプリケーションを起動する:
```bash
streamlit run app.py
```

ブラウザが自動的に開き、インターフェースが表示されます。

### ルールベース生成

1. プリセットコード進行を選択するか、カスタムコード進行を入力
2. スタイル（スタンダード/ビバップ/モーダル/フュージョン）を選択
3. テンポ、小節数、オクターブなどのパラメータを調整
4. 「ソロを生成」ボタンをクリック
5. 生成されたソロを楽譜で確認し、試聴する
6. MIDIファイルをダウンロードして、DAWソフトなどで編集・利用する

### MIDIデータセット処理（開発者向け）

大規模なMIDIファイルからコード進行を抽出するには:

1. MIDIファイルを `jazz_ai_dataset/raw_midis` ディレクトリに配置
2. 以下のコマンドを実行:
```bash
python src/ml/extract_chords.py
```
3. 抽出結果は `jazz_ai_dataset/processed` ディレクトリに保存されます

## 機械学習モデルの開発（進行中）

現在、以下の機械学習機能を開発中です:

1. 抽出したコード進行からジャズソロを生成するモデル
2. スタイルに基づいたソロ生成
3. APIベースの機械学習モデル統合

独自のジャズソロデータを使ってモデルを訓練するためのツールも開発中です。

## プロジェクト構成

```
jazz-solo-generator/
├── app.py                   # メインアプリケーション
├── src/                     # ソースコードディレクトリ
│   ├── constants.py         # 定数や設定値
│   ├── theory.py            # 音楽理論関連の関数
│   ├── patterns.py          # リズムパターンと生成
│   ├── generator.py         # ルールベースソロ生成
│   ├── audio_utils.py       # 音声処理関連の関数
│   ├── visualization.py     # 楽譜表示など
│   └── ml/                  # 機械学習関連
│       ├── extract_chords.py  # コード進行抽出ツール
│       ├── data_preparation.py  # データ前処理
│       ├── model.py         # モデル定義と訓練
│       ├── generator.py     # ML生成関数
│       └── train.py         # 訓練スクリプト
├── models/                  # 訓練済みモデル保存場所
├── jazz_midis/              # 訓練用MIDIファイル
├── jazz_ai_dataset/         # 大規模データセット
│   ├── raw_midis/           # 生のMIDIファイル
│   └── processed/           # 処理済みファイル
├── requirements.txt         # 必要ライブラリリスト
└── README.md                # このファイル
```

## 技術的詳細

### ルールベース生成機能

- コード進行の解析と対応するスケールの選択
- ジャズスタイル別のリズムパターン生成
- コードトーンの優先と適切なクロマチック音の追加
- スタイル固有の特徴（シンコペーション、音域など）の適用

### 機械学習生成機能（開発中）

- LSTM/Transformer ニューラルネットワークによる音符シーケンスの学習
- 実際のジャズミュージシャンの演奏パターンから学習
- FastAPI ベースの推論サービス
- ルールベースと機械学習を組み合わせたハイブリッドアプローチ

## 今後の拡張予定

- ルールベースと機械学習を統合したハイブリッドソロ生成
- ドラムパターンの生成と追加
- 楽譜エディタ機能
- ミュージシャン特化型の機械学習モデル
- 和声的コンテキストを考慮した生成
- ユーザーフィードバックに基づくモデル改善機能

## 貢献方法

プロジェクトへの貢献を歓迎します。以下の方法で貢献できます:

1. 機能追加や修正のための Pull Request
2. バグ報告や機能提案のための Issue
3. ドキュメントの改善
4. MIDIデータセットの提供（学習目的）

## ライセンス

MITライセンス

## 謝辞

- FluidSynthチーム
- music21ライブラリの開発者
- Streamlitフレームワークの開発者
- TensorFlow/PyTorchコミュニティ
- このプロジェクトのために音楽データを提供してくださったミュージシャンの皆様
