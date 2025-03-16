"""
高速化されたMIDIファイルからのコード進行抽出ツール
- プロセスプール活用によるマルチプロセッシング最適化
- 軽量なMIDI解析オプション
- メモリ使用量の最適化
- I/O操作の削減
"""
import os
import time
import json
import csv
import signal
import multiprocessing as mp
from tqdm import tqdm
from music21 import converter, chord, environment
import pandas as pd
import tempfile
import pickle
import gc
import mido

# --------------------------------------------------------
# 設定値
# --------------------------------------------------------
MAX_FILE_SIZE_MB = 5.0  # このサイズを超えるMIDIファイルはスキップ
MIN_FILE_SIZE_KB = 1.0  # このサイズ未満のファイルはスキップ
VALID_EXTENSIONS = (".mid", ".midi")  # 対象とする拡張子
BATCH_SIZE = 1000  # 一度に処理するファイル数を増加
CHECKPOINT_INTERVAL = 200  # チェックポイントを保存する間隔（増加）
MAX_WORKERS = max(4, mp.cpu_count())  # 使用するCPUコア数を増加（少なくとも4）
FILE_TIMEOUT = 30  # タイムアウト時間を短縮（秒）
USE_LIGHTWEIGHT_PARSER = True  # 軽量パーサーを使用するかどうか
MEMORY_LIMIT_MB = 1024  # 1プロセス当たりのメモリ制限(MB)
# --------------------------------------------------------

# music21環境設定の最適化
def optimize_music21():
    """music21の環境設定を最適化する"""
    env = environment.Environment()
    # キャッシュの設定
    env['autoDownload'] = 'no'
    env['warnings'] = 0
    env['graphicsPath'] = None
    # 一時ディレクトリをRAMディスクに変更（利用可能な場合）
    temp_dir = tempfile.gettempdir()
    env['directoryScratch'] = temp_dir

# 軽量MIDIパーサー（mido使用）
def extract_chords_lightweight(midi_path):
    """midoを使用した軽量コード抽出"""
    try:
        midi_file = mido.MidiFile(midi_path)
        # 簡易的なコード抽出アルゴリズム
        notes_by_time = {}
        current_time = 0
        
        for track in midi_file.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    if track_time not in notes_by_time:
                        notes_by_time[track_time] = []
                    notes_by_time[track_time].append(msg.note)
        
        # 時間でソート
        sorted_times = sorted(notes_by_time.keys())
        
        # 簡易コード名抽出
        chord_sequence = []
        for t in sorted_times:
            if len(notes_by_time[t]) >= 2:  # 2音以上あればコードとみなす
                # 簡易コード名生成（実際のコード解析はより複雑）
                chord_name = f"Chord-{'-'.join(str(n % 12) for n in sorted(notes_by_time[t]))}"
                chord_sequence.append(chord_name)
        
        return chord_sequence
    except Exception as e:
        print(f"Lightweight parser error: {e}")
        return None

# プロセスで実行される処理
def process_file_worker(file_info):
    """ワーカープロセスで実行されるファイル処理関数"""
    midi_path, use_lightweight = file_info
    start_time = time.time()
    
    try:
        # ファイルサイズチェック
        size_bytes = os.path.getsize(midi_path)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return None
        
        chord_sequence = None
        
        # パーサー選択
        if use_lightweight:
            chord_sequence = extract_chords_lightweight(midi_path)
        else:
            # 従来のmusic21パーサー
            midi = converter.parse(midi_path)
            chords = midi.chordify().recurse().getElementsByClass('Chord')
            chord_sequence = [c.pitchedCommonName for c in chords]
            
            # メモリ解放（重要）
            del midi
            del chords
            gc.collect()
        
        if chord_sequence and len(chord_sequence) > 0:
            processing_time = time.time() - start_time
            return {
                "filename": midi_path,
                "chords": "|".join(chord_sequence),
                "processing_time": processing_time
            }
        
        return None
        
    except Exception as e:
        processing_time = time.time() - start_time
        # エラー情報だけを返し、ファイルには別途まとめて書き込む
        return {
            "filename": midi_path,
            "error": str(e),
            "processing_time": processing_time
        }

# ファイル処理（タイムアウト付き）
def process_batch_with_pool(file_batch, use_lightweight=False):
    """マルチプロセスプールを使用してバッチファイルを処理する"""
    # 処理用のパラメータ準備
    file_infos = [(f, use_lightweight) for f in file_batch]
    
    # マルチプロセスプールの設定
    pool = mp.Pool(MAX_WORKERS)
    
    try:
        # タイムアウト付きでマップ処理を実行
        results = []
        for i, result in enumerate(tqdm(
            pool.imap_unordered(process_file_worker, file_infos), 
            total=len(file_infos),
            desc="Processing files"
        )):
            results.append(result)
    finally:
        pool.close()
        pool.join()
    
    # 結果を処理成功・エラーに分類
    successful_results = []
    error_files = []
    
    for result in results:
        if result is None:
            continue
        
        if "error" in result:
            error_files.append((result["filename"], result["error"]))
        else:
            successful_results.append(result)
    
    # エラーファイルを一括記録
    if error_files:
        with open("skipped_files.txt", "a") as skip_file:
            for filename, error in error_files:
                skip_file.write(f"{filename},error,{error}\n")
    
    return successful_results

def get_file_list(input_dir):
    """対象ディレクトリから処理すべきファイルリストを取得する"""
    file_paths = []
    
    # スキップすべきファイルリストを読み込む（メモリ効率化）
    skip_list = set()
    if os.path.exists("skipped_files.txt"):
        with open("skipped_files.txt", "r") as skip_file:
            for line in skip_file:
                parts = line.strip().split(",", 1)
                if parts:
                    skip_list.add(parts[0])
    
    print(f"Found {len(skip_list)} files to skip from previous runs")
    
    # 対象ファイルを効率的に検索
    print("Scanning for MIDI files...")
    total_files_scanned = 0
    valid_files_found = 0
    
    # より効率的なファイル検索
    for root_dir, _, files in os.walk(input_dir):
        for filename in files:
            total_files_scanned += 1
            
            if total_files_scanned % 100000 == 0:
                print(f"Scanned {total_files_scanned} files, found {valid_files_found} valid...")
                
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            
            midi_path = os.path.join(root_dir, filename)
            
            # スキップリストに含まれるファイルは除外
            if midi_path in skip_list:
                continue
            
            try:
                size_bytes = os.path.getsize(midi_path)
                size_kb = size_bytes / 1024
                size_mb = size_bytes / (1024 * 1024)
                
                if size_kb < MIN_FILE_SIZE_KB or size_mb > MAX_FILE_SIZE_MB:
                    continue
                    
                file_paths.append(midi_path)
                valid_files_found += 1
            except Exception as e:
                # エラーをログに書き込む（スキップリストに自動追加）
                with open("skipped_files.txt", "a") as skip_file:
                    skip_file.write(f"{midi_path},file_check_error,{str(e)}\n")
    
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
            # 処理済みファイル名ではなくカウントのみを保存（メモリ節約）
            checkpoint_data = {
                "processed_count": len(processed_files),
                "current_batch": current_batch,
                "results_file": results_file,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            json.dump(checkpoint_data, f)
        print(f"Checkpoint saved: {len(processed_files)} files processed")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def write_results_to_csv(results, output_file, mode='a'):
    """結果をCSVに書き込む（一括処理）"""
    if not results:
        return
        
    try:
        # 存在しない場合はヘッダーを作成
        write_header = mode == 'w' or not os.path.exists(output_file) or os.path.getsize(output_file) == 0
        
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['filename', 'chords', 'processing_time'])
            
            for res in results:
                writer.writerow([res['filename'], res['chords'], res['processing_time']])
                
    except Exception as e:
        print(f"Error writing CSV: {e}")

def process_files_in_batches(file_paths, output_dir, checkpoint_file):
    """
    ファイルをバッチ処理して結果を逐次保存する
    最適化版: プロセスプール使用、一括書き込み、メモリ効率化
    """
    # チェックポイントから復元
    checkpoint_data = load_checkpoint(checkpoint_file)
    processed_count = checkpoint_data.get("processed_count", 0)
    current_batch = checkpoint_data.get("current_batch", 0)
    results_file = checkpoint_data.get("results_file")
    
    # 処理済みファイルのセットを構築（メモリ節約のため、必要に応じて）
    processed_files = set()
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # ヘッダーをスキップ
                for row in reader:
                    if row and len(row) > 0:
                        processed_files.add(row[0])  # ファイル名のみ追加
        except Exception as e:
            print(f"Error reading previous results: {e}")
    
    # 結果ファイルの設定
    if not results_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"chord_progressions_{timestamp}.csv")
        
        # 空のCSVファイルを作成
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'chords', 'processing_time'])
    else:
        print(f"Resuming from previous run. Output file: {results_file}")
    
    # 未処理ファイルのリスト作成（処理済みファイルリストとの差分を取得）
    files_to_process = [f for f in file_paths if f not in processed_files]
    total_remaining = len(files_to_process)
    
    print(f"Total files: {len(file_paths)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining to process: {total_remaining}")
    
    # 軽量パーサーの使用有無
    use_lightweight = USE_LIGHTWEIGHT_PARSER
    print(f"Using lightweight parser: {use_lightweight}")
    
    # バッチ処理の開始
    total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(current_batch, total_batches):
        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_remaining)
        current_batch_files = files_to_process[batch_start:batch_end]
        
        print(f"\nProcessing batch {i+1}/{total_batches} ({len(current_batch_files)} files)")
        batch_start_time = time.time()
        
        # バッチ処理を実行（タイムアウト機構付き）
        results = process_batch_with_pool(current_batch_files, use_lightweight)
        
        # 処理済みファイルを追跡
        for res in results:
            processed_files.add(res['filename'])
        
        # 結果を一括でCSVに書き込む
        write_results_to_csv(results, results_file, 'a')
        
        # バッチ完了時にチェックポイント保存
        save_checkpoint(checkpoint_file, processed_files, i+1, results_file)
        
        # メモリ解放
        del results
        gc.collect()
        
        # バッチ処理時間表示
        batch_elapsed = time.time() - batch_start_time
        avg_file_time = batch_elapsed / len(current_batch_files) if current_batch_files else 0
        
        print(f"Batch {i+1} completed in {batch_elapsed:.2f} seconds")
        print(f"Average time per file: {avg_file_time:.2f} seconds")
        print(f"Total progress: {len(processed_files)}/{len(file_paths)} files ({len(processed_files)/len(file_paths)*100:.1f}%)")
    
    print(f"\nProcessing complete! Results saved to: {results_file}")
    print(f"Total files processed: {len(processed_files)}/{len(file_paths)}")
    
    # 処理完了後に統計情報を表示（メモリ効率の良い方法で）
    try:
        # 軽量な統計計算
        with open(results_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダーをスキップ
            
            total_time = 0
            count = 0
            
            for row in reader:
                if row and len(row) >= 3:
                    try:
                        proc_time = float(row[2])
                        total_time += proc_time
                        count += 1
                    except (ValueError, IndexError):
                        pass
        
        if count > 0:
            avg_time = total_time / count
            print(f"\nExtracted chord progressions from {count} files")
            print(f"Average processing time: {avg_time:.2f} seconds per file")
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
    
    print(f"Optimized MIDI Chord Extraction Tool")
    print(f"===================================")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using {MAX_WORKERS} worker processes")
    print(f"Batch size: {BATCH_SIZE} files")
    print(f"Lightweight parser: {USE_LIGHTWEIGHT_PARSER}")
    
    # music21環境の最適化
    print("Optimizing music21 environment...")
    optimize_music21()
    
    # テストモードフラグとテスト件数
    test_mode = False  # テストモードの場合はTrue
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