import os
from music21 import converter, chord
import pandas as pd

# 現在のスクリプトがあるディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# プロジェクトルート（ここでは2階層上にあると仮定）
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 入力MIDIファイルが格納されているディレクトリ
input_dir = os.path.join(project_root, "jazz_ai_dataset", "raw_midis")

# 出力CSVファイルのパス（前処理済みデータ保存用）
output_csv = os.path.join(project_root, "jazz_ai_dataset", "processed", "chord_progressions.csv")

# processedディレクトリがなければ作成
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

data = []

# os.walk() を使って再帰的にファイルを探索
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith((".mid", ".midi")):
            midi_path = os.path.join(root, filename)
            try:
                midi = converter.parse(midi_path)
                # MIDIをコードに変換（chordify）
                chords = midi.chordify().recurse().getElementsByClass('Chord')
                chord_sequence = [c.pitchedCommonName for c in chords]
                data.append({
                    "filename": midi_path,
                    "chords": "|".join(chord_sequence)
                })
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")

# DataFrameに変換してCSVに保存
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"Chord progressions saved to: {output_csv}")
