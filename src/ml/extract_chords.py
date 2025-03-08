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
processed_dir = os.path.dirname(output_csv)
os.makedirs(processed_dir, exist_ok=True)

data = []

# input_dir内のMIDIファイルを処理
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".mid", ".midi")):
        midi_path = os.path.join(input_dir, filename)
        try:
            midi = converter.parse(midi_path)
            # MIDIをコードに変換（chordify）
            chords = midi.chordify().recurse().getElementsByClass('Chord')
            chord_sequence = [c.pitchedCommonName for c in chords]
            data.append({
                "filename": filename,
                "chords": "|".join(chord_sequence)
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# DataFrameに変換してCSVに保存
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"Chord progressions saved to: {output_csv}")
