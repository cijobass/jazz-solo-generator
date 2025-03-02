# ジャズアドリブジェネレーター

ジャズのコード進行に基づいてアドリブソロを自動生成するStreamlitアプリケーションです。

## 機能

- コード進行の入力（カスタムまたはプリセット）
- ソロラインとベースラインの生成
- 生成されたメロディのMIDIファイル出力
- ブラウザ内での試聴機能
- 楽譜表示機能
- テンポ、オクターブなどのカスタマイズオプション

## インストール方法

### 前提条件
- Python 3.10以上
- FluidSynth（オーディオ再生用）
- SoundFontファイル（例：FluidR3_GM_GS.sf2）

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

## 使用方法

アプリケーションを起動する:
```bash
streamlit run app.py
```

ブラウザが自動的に開き、インターフェースが表示されます。

1. プリセットコード進行を選択するか、カスタムコード進行を入力
2. テンポ、小節数、オクターブなどのパラメータを調整
3. 「ソロを生成」ボタンをクリック
4. 生成されたソロを楽譜で確認し、試聴する
5. MIDIファイルをダウンロードして、DAWソフトなどで編集・利用する

## プロジェクト構成

```
jazz-solo-generator/
├── src/                   # ソースコードディレクトリ
│   ├── __init__.py        # Pythonパッケージとして認識させるファイル
│   ├── generator.py       # ジャズソロ生成機能
│   ├── audio_utils.py     # 音声処理関連の関数
│   ├── visualization.py   # 楽譜表示など視覚化関連の関数
│   └── constants.py       # 定数や設定値
├── app.py                 # メインアプリケーション
├── requirements.txt       # 必要なライブラリのリスト
└── README.md              # このファイル
```

## 今後の拡張予定

- より高度なスタイル別の生成アルゴリズム
- 機械学習を用いた実際のジャズソロの学習と再現
- ドラムパターンの生成と追加
- 楽譜エディタ機能

## ライセンス

MITライセンス

## 謝辞

- FluidSynthチーム
- music21ライブラリの開発者
- Streamlitフレームワークの開発者
