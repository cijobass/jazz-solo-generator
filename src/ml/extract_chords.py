"""
改良版MIDIファイルからのコード進行抽出ツール
- バッチ処理＋チェックポイント機能
- Windows環境最適化
- メモリ効率の向上
"""
import os
import time
import json
import csv
import multiprocessing as mp
from tqdm import tqdm
from music21 import converter, chord
import pandas as pd

# --------------------------------------------------------
# 設定値
# --------------------------------------------------------
MAX_FILE_SIZE_MB = 5.0  # このサイズを超えるMIDIファイルはスキップ
MIN_FILE_SIZE_KB = 1.0  # このサイズ未満のファイルはスキップ
VALID_EXTENSIONS = (".mid", ".midi")  # 対象とする拡張子
BATCH_SIZE = 200  # 一度に処理するファイル数
CHECKPOINT_INTERVAL = 50  # チェックポイントを保存する間隔（ファイル数）
MAX_WORKERS = max(1, mp.cpu_count() - 1)  # CPUコア数 - 1 (少なくとも1)
# --------------------------------------------------------

def process_file(midi_path):
    """
    1ファイルのMIDIを解析してコード進行を抽出する関数。
    エラーがあればNoneを返す。
    """
    try:
        start_time = time.time()
        
        # ファイルサイズチェック（念のため）
        size_bytes = os.path.getsize(midi_path)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return None
            
        midi = converter.parse(midi_path)
        
        # MIDIをコードに変換（chordify）してChordオブジェクトを取得
        chords = midi.chordify().recurse().getElementsByClass('Chord')
        
        # コード進行の抽出（空なら None を返す）
        chord_sequence = [c.pitchedCommonName for c in chords]
        if not chord_sequence:
            return None
            
        elapsed = time.time() - start_time
        print(f"Processed {midi_path} in {elapsed:.2f} seconds")
        
        return {
            "filename": midi_path,
            "chords": "|".join(chord_sequence),
            "processing_time": elapsed
        }
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None

def get_file_list(input_dir):
    """
    対象ディレクトリから処理すべきファイルリストを取得する
    ファイルサイズでのフィルタリングを含む
    """
    file_paths = []
    
    print("Scanning for MIDI files...")
    total_files_scanned = 0
    
    for root_dir, _, files in os.walk(input_dir):
        for filename in files:
            total_files_scanned += 1
            
            # 進捗表示（大量ファイルのスキャン時）
            if total_files_scanned % 10000 == 0:
                print(f"Scanned {total_files_scanned} files...")
                
            # 拡張子チェック
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            
            midi_path = os.path.join(root_dir, filename)
            
            # ファイルサイズチェック
            try:
                size_bytes = os.path.getsize(midi_path)
                size_kb = size_bytes / 1024
                size_mb = size_bytes / (1024 * 1024)
                
                if size_kb < MIN_FILE_SIZE_KB or size_mb > MAX_FILE_SIZE_MB:
                    continue
                    
                file_paths.append(midi_path)
            except Exception as e:
                print(f"Error checking file {midi_path}: {e}")
    
    print(f"Found {len(file_paths)} valid MIDI files out of {total_files_scanned} scanned")
    return file_paths

def load_checkpoint(checkpoint_file):
    """チェックポイントファイルを読み込む"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                return checkpoint_data
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {"processed_files": [], "current_batch": 0, "results_file": None}
    return {"processed_files": [], "current_batch": 0, "results_file": None}

def save_checkpoint(checkpoint_file, processed_files, current_batch, results_file):
    """チェックポイントファイルを保存する"""
    try:
        with open(checkpoint_file, 'w') as f:
            checkpoint_data = {
                "processed_files": processed_files,
                "current_batch": current_batch,
                "results_file": results_file,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            json.dump(checkpoint_data, f)
        print(f"Checkpoint saved: {len(processed_files)} files processed")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def process_files_in_batches(file_paths, output_dir, checkpoint_file):
    """
    ファイルをバッチ処理して結果を逐次保存する
    チェックポイント機能付き
    """
    # チェックポイントから復元
    checkpoint_data = load_checkpoint(checkpoint_file)
    processed_files = set(checkpoint_data["processed_files"])
    current_batch = checkpoint_data["current_batch"]
    results_file = checkpoint_data["results_file"]
    
    # 結果ファイルの設定
    if not results_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"chord_progressions_{timestamp}.csv")
        
        # ヘッダー行を書き出し
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'chords', 'processing_time'])
    else:
        print(f"Resuming from previous run. Output file: {results_file}")
    
    # 未処理ファイルのリスト作成
    files_to_process = [f for f in file_paths if f not in processed_files]
    total_remaining = len(files_to_process)
    
    print(f"Total files: {len(file_paths)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining to process: {total_remaining}")
    
    # バッチ処理の開始
    total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(current_batch, total_batches):
        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_remaining)
        current_batch_files = files_to_process[batch_start:batch_end]
        
        print(f"\nProcessing batch {i+1}/{total_batches} ({len(current_batch_files)} files)")
        batch_start_time = time.time()
        
        # 並列処理
        results = []
        with mp.Pool(MAX_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(process_file, current_batch_files), 
                             total=len(current_batch_files), 
                             desc=f"Batch {i+1}/{total_batches}"):
                if result is not None:
                    results.append(result)
                    
                    # 一定間隔でチェックポイント更新（細かめに）
                    if len(results) % CHECKPOINT_INTERVAL == 0:
                        # 中間結果のCSV追加
                        with open(results_file, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            for res in results[-CHECKPOINT_INTERVAL:]:
                                writer.writerow([res['filename'], res['chords'], res['processing_time']])
                        
                        # 処理済みリストの更新
                        for res in results[-CHECKPOINT_INTERVAL:]:
                            processed_files.add(res['filename'])
                            
                        # チェックポイント保存
                        save_checkpoint(checkpoint_file, list(processed_files), i, results_file)
        
        # バッチ内の残りの結果をCSVに追加
        remaining_results = [r for r in results if r['filename'] not in processed_files]
        if remaining_results:
            with open(results_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for res in remaining_results:
                    writer.writerow([res['filename'], res['chords'], res['processing_time']])
                    processed_files.add(res['filename'])
        
        # バッチ完了時のチェックポイント保存
        save_checkpoint(checkpoint_file, list(processed_files), i+1, results_file)
        
        # バッチ処理時間表示
        batch_elapsed = time.time() - batch_start_time
        print(f"Batch {i+1} completed in {batch_elapsed:.2f} seconds")
        print(f"Extracted chords from {len(results)}/{len(current_batch_files)} files in this batch")
        print(f"Total progress: {len(processed_files)}/{len(file_paths)} files ({len(processed_files)/len(file_paths)*100:.1f}%)")
    
    print(f"\nProcessing complete! Results saved to: {results_file}")
    print(f"Total files processed: {len(processed_files)}/{len(file_paths)}")
    
    # 処理完了後に統計情報を表示
    try:
        df = pd.read_csv(results_file)
        print(f"\nExtracted chord progressions from {len(df)} files")
        print(f"Average processing time: {df['processing_time'].mean():.2f} seconds per file")
    except Exception as e:
        print(f"Error generating statistics: {e}")
    
    return results_file

def main():
    """メイン処理関数"""
    # 現在のスクリプトディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトルートを2階層上に設定
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 入出力ディレクトリの設定
    input_dir = os.path.join(project_root, "jazz_ai_dataset", "raw_midis")
    output_dir = os.path.join(project_root, "jazz_ai_dataset", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # チェックポイントファイルの設定
    checkpoint_file = os.path.join(output_dir, "extraction_checkpoint.json")
    
    print(f"MIDI Chord Extraction Tool")
    print(f"==========================")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using {MAX_WORKERS} worker processes")
    print(f"Batch size: {BATCH_SIZE} files")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} files")
    
    # テストモードフラグとテスト件数
    test_mode = True  # テストモードの場合はTrue
    test_file_count = 1000  # テストするファイル数
    
    # 処理開始時間
    start_all = time.time()
    
    # ファイルリストの取得
    file_paths = get_file_list(input_dir)
    if not file_paths:
        print("No MIDI files found. Check the input directory.")
        return
    
    # テストモードの場合は処理するファイル数を制限
    if test_mode:
        original_count = len(file_paths)
        file_paths = file_paths[:min(test_file_count, original_count)]
        print(f"TEST MODE: Processing only {len(file_paths)} of {original_count} files")
    
    # バッチ処理実行
    output_file = process_files_in_batches(file_paths, output_dir, checkpoint_file)
    
    # 総処理時間
    total_elapsed = time.time() - start_all
    print(f"\nTotal processing time: {total_elapsed:.2f} seconds")
    print(f"Average time per file: {total_elapsed/len(file_paths):.2f} seconds")
    print(f"\nOutput file: {output_file}")

if __name__ == '__main__':
    # Windows環境でのマルチプロセッシング対応
    mp.freeze_support()
    main()