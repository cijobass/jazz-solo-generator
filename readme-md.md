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
- 機械学習を用いた実際のジャズソロの学習と再現

## インストール方法

### 前提条件
- Python 3.10以上
- FluidSynth（オーディオ再生用）
- SoundFontファイル（例：FluidR3_GM_GS.sf2）
- （オプション）LilyPond（高品質な楽譜表示用）
- （機械学習機能用）TensorFlow 2.x

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

6. 機械学習機能のセットアップ（オプション）
   - `jazz_midis`ディレクトリにジャズソロのMIDIファイルを配置
   - モデルを訓練する
   ```bash
   python -m src.ml.train
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

### 機械学習ベース生成

1. 「機械学習モデル」スタイルを選択
2. 温度パラメータ（創造性）と生成する音符数を設定
3. 「ソロを生成」ボタンをクリック
4. 生成されたソロを楽譜で確認し、試聴する
5. MIDIファイルをダウンロードして、DAWソフトなどで編集・利用する

## 機械学習モデルの訓練

独自のジャズソロデータを使ってモデルを訓練するには：

1. `jazz_midis`ディレクトリにジャズソロのMIDIファイルを配置
   - 著名なジャズミュージシャン（Charlie Parker, Miles Davis, John Coltrane など）のソロが適しています
   - 無料のMIDIファイルは[midiworld.com](https://www.midiworld.com/jazz.htm)などで入手可能です

2. モデルを訓練する
```bash
python -m src.ml.train
```

3. 訓練済みモデルは`models`ディレクトリに保存されます

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
│       ├── data_preparation.py  # データ前処理
│       ├── model.py         # モデル定義と訓練
│       ├── generator.py     # ML生成関数
│       └── train.py         # 訓練スクリプト
├── models/                  # 訓練済みモデル保存場所
├── jazz_midis/              # 訓練用MIDIファイル
├── requirements.txt         # 必要ライブラリリスト
└── README.md                # このファイル
```

## 技術的詳細

### ルールベース生成機能

- コード進行の解析と対応するスケールの選択
- ジャズスタイル別のリズムパターン生成
- コードトーンの優先と適切なクロマチック音の追加
- スタイル固有の特徴（シンコペーション、音域など）の適用

### 機械学習生成機能

- LSTM（Long Short-Term Memory）ニューラルネットワークによる音符シーケンスの学習
- 実際のジャズミュージシャンの演奏パターンから学習
- 「温度」パラメータによる生成の創造性調整
- 学習済みモデルを使用した新しいソロの生成

## 今後の拡張予定

- より高度なスタイル別の生成アルゴリズム
- ドラムパターンの生成と追加
- 楽譜エディタ機能
- ミュージシャン特化型の機械学習モデル
- 和声的コンテキストを考慮した生成

## ライセンス

MITライセンス

## 謝辞

- FluidSynthチーム
- music21ライブラリの開発者
- Streamlitフレームワークの開発者
- TensorFlowチーム
- このプロジェクトのために音楽データを提供してくださったミュージシャンの皆様
