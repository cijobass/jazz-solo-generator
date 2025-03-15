import os
import time
import multiprocessing as mp
from music21 import converter, chord
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------
# 設定値（必要に応じて調整）
# --------------------------------------------------------
MAX_FILE_SIZE_MB = 5.0  # このサイズを超えるMIDIファイルはスキップ (例: 5MB)
MIN_FILE_SIZE_KB = 1.0  # このサイズ未満のファイルはスキップ (例: 1KB)
VALID_EXTENSIONS = (".mid", ".midi")  # 対象とする拡張子
# --------------------------------------------------------

def process_file(midi_path):
    """
    1ファイルのMIDIを解析してコード進行を抽出する関数。
    エラーがあればNoneを返す。
    """
    start_time = time.time()
    try:
        midi = converter.parse(midi_path)
        # MIDIをコードに変換（chordify）してChordオブジェクトを取得
        chords = midi.chordify().recurse().getElementsByClass('Chord')
        chord_sequence = [c.pitchedCommonName for c in chords]
        elapsed = time.time() - start_time
        print(f"Processed {midi_path} in {elapsed:.2f} seconds")
        return {"filename": midi_path, "chords": "|".join(chord_sequence)}
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None

def main():
    # 現在のスクリプトディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトルートを2階層上に設定
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 入力MIDIファイルのディレクトリ
    input_dir = os.path.join(project_root, "jazz_ai_dataset", "raw_midis")
    # 出力CSVファイルのパス（processedフォルダが存在しなければ自動作成）
    output_csv = os.path.join(project_root, "jazz_ai_dataset", "processed", "chord_progressions.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 再帰的に全MIDIファイルのパスを取得（フィルタリング込み）
    file_paths = []
    for root_dir, dirs, files in os.walk(input_dir):
        for filename in files:
            # 拡張子チェック
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            
            midi_path = os.path.join(root_dir, filename)
            # ファイルサイズチェック
            size_bytes = os.path.getsize(midi_path)
            size_mb = size_bytes / (1024 * 1024)
            size_kb = size_bytes / 1024
            
            # 不要ファイルのフィルタリング例
            if size_kb < MIN_FILE_SIZE_KB or size_mb > MAX_FILE_SIZE_MB:
                continue
            
            file_paths.append(midi_path)
    
    total_files = len(file_paths)
    print(f"Found {total_files} MIDI files after filtering.")
    
    start_all = time.time()
    
    # 並列処理と進捗表示：mp.Pool と tqdm の組み合わせ
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, file_paths), total=total_files, desc="Processing MIDI files"):
            results.append(result)
    
    # Noneを除外
    data = [res for res in results if res is not None]
    
    # DataFrameに変換してCSVに保存
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    total_elapsed = time.time() - start_all
    print(f"Processed {total_files} files in {total_elapsed:.2f} seconds")
    print(f"Chord progressions saved to: {output_csv}")

if __name__ == '__main__':
    main()
