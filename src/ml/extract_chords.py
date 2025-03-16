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
import sys

# CSVフィールドサイズ制限を引き上げる（Windows環境に対応）
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # Windows環境では sys.maxsize が大きすぎるため、適切な値を設定
    # 2^31 - 1 = 2147483647 (32ビットプラットフォームの最大値)
    csv.field_size_limit(2147483647)
try:
    import mido  # 追加: 軽量なMIDIパーサー
    MIDO_AVAILABLE = True
except ImportError:
    print("警告: midoパッケージがインストールされていません。")
    print("軽量パーサーが使用できないため、従来のmusic21パーサーを使用します。")
    print("インストールするには: pip install mido")
    MIDO_AVAILABLE = False

# --------------------------------------------------------
# 設定値
# --------------------------------------------------------
MAX_FILE_SIZE_MB = 5.0  # このサイズを超えるMIDIファイルはスキップ
MIN_FILE_SIZE_KB = 1.0  # このサイズ未満のファイルはスキップ
VALID_EXTENSIONS = (".mid", ".midi")  # 対象とする拡張子
BATCH_SIZE = 1000  # 一度に処理するファイル数を増加
CHECKPOINT_INTERVAL = 200  # チェックポイントを保存する間隔（増加）
MAX_WORKERS = min(16, max(4, mp.cpu_count() - 2))  # CPUコア数を基に適切なワーカー数に制限（最大16、最小4）
FILE_TIMEOUT = 30  # タイムアウト時間を短縮（秒）
USE_LIGHTWEIGHT_PARSER = True  # 軽量パーサーを使用するかどうか（初期設定値）
MEMORY_LIMIT_MB = 1024  # 1プロセス当たりのメモリ制限(MB)
# --------------------------------------------------------

# music21環境設定の最適化
def optimize_music21():
    """music21の環境設定を最適化する"""
    try:
        env = environment.Environment()
        # キャッシュの設定（正しい値を使用）
        env['autoDownload'] = 'allow'  # 'no'ではなく'allow'または'deny'を使用
        env['warnings'] = 0
        
        # グラフィックパスを無効化
        try:
            env['graphicsPath'] = None
        except:
            pass  # このオプションが利用できない場合は無視
            
        # 一時ディレクトリをRAMディスクに変更（利用可能な場合）
        temp_dir = tempfile.gettempdir()
        env['directoryScratch'] = temp_dir
        
        print("music21環境設定を最適化しました")
    except Exception as e:
        print(f"music21環境設定の最適化中にエラーが発生しました: {e}")
        print("処理は続行します...")

# 軽量MIDIパーサー（mido使用）
def extract_chords_lightweight(midi_path):
    """midoを使用した軽量コード抽出"""
    try:
        if not MIDO_AVAILABLE:
            raise ImportError("mido パッケージがインストールされていません")
            
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
        
        if not chord_sequence:
            return None
            
        return chord_sequence
    except Exception as e:
        print(f"軽量パーサーエラー: {midi_path}: {e}")
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
                print(f"チェックポイントファイルから読み込み: {checkpoint_file}")
                print(f"最終更新: {checkpoint_data.get('last_updated', '不明')}")
                print(f"バッチ位置: {checkpoint_data.get('current_batch', 0)}")
                return checkpoint_data
        except Exception as e:
            print(f"チェックポイント読み込みエラー: {e}")
            print("新規チェックポイントで処理を開始します")
            return {"processed_count": 0, "current_batch": 0, "results_file": None}
    return {"processed_count": 0, "current_batch": 0, "results_file": None}

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
    
    # 古い形式のチェックポイントファイルへの対応
    if "processed_count" in checkpoint_data:
        processed_count = checkpoint_data["processed_count"]
        print(f"チェックポイントから処理済み件数を読込: {processed_count}件")
    elif "processed_files" in checkpoint_data and isinstance(checkpoint_data["processed_files"], list):
        processed_count = len(checkpoint_data["processed_files"])
        print(f"チェックポイントから処理済みファイルリストを読込: {processed_count}件")
    else:
        processed_count = 0
        print("処理済み件数の情報がチェックポイントにありません")
        
    current_batch = checkpoint_data.get("current_batch", 0)
    results_file = checkpoint_data.get("results_file")
    
    # 処理済みファイルのセットを構築（メモリ節約のため、必要に応じて）
    processed_files = set()
    estimated_count = 0
    
    if results_file and os.path.exists(results_file):
        try:
            # 行数をカウントして処理済みファイル数を推定
            def count_lines(filename):
                count = 0
                with open(filename, 'rb') as f:
                    # バイナリモードで読み込み、改行文字をカウント
                    for _ in f:
                        count += 1
                return count - 1  # ヘッダー行を除く
            
            estimated_count = count_lines(results_file)
            print(f"CSV行数から推定した処理済みファイル数: {estimated_count}件")
            
            # チェックポイントと推定値に大きな差がある場合は調整
            if processed_count == 0 or abs(processed_count - estimated_count) > 1000:
                print(f"チェックポイント({processed_count})と推定値({estimated_count})に差があります")
                processed_count = estimated_count
                print(f"処理済み件数を推定値({estimated_count})に調整しました")
            
            # 大きなファイルでもサンプル数を制限
            max_samples = min(10000, processed_count)
            if max_samples > 0 and processed_count > 0:
                sample_rate = max(1, processed_count // max_samples)
                print(f"処理済みファイル名のサンプリング: {max_samples}件（サンプリング率: 1/{sample_rate}）")
                
                # 効率的なサンプリング
                with open(results_file, 'r', newline='', encoding='utf-8') as f:
                    # 最初の行（ヘッダー）をスキップ
                    next(f, None)
                    
                    line_num = 0
                    for line in f:
                        line_num += 1
                        if line_num % sample_rate == 0:
                            try:
                                # 最初のカンマまでの部分だけを取得（ファイル名）
                                filename = line.split(',', 1)[0].strip('"\'')
                                if filename:
                                    processed_files.add(filename)
                            except (IndexError, ValueError):
                                continue  # 問題のある行はスキップ
                            
                            # 十分なサンプルを収集したら終了
                            if len(processed_files) >= max_samples:
                                break
                
                print(f"サンプリングされた処理済みファイル: {len(processed_files)}件")
                
                # 処理済みファイル数をCSV行数から取得
                if processed_count == 0:
                    processed_count = estimated_count
            
            print(f"以前の実行から{processed_count}ファイルの処理記録を読み込みました")
        except Exception as e:
            print(f"以前の結果の読み込みエラー: {e}")
            print("処理を新規開始します")
    
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
    
    # バッチ数を計算
    total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"総ファイル数: {len(file_paths)}")
    print(f"処理済み: {len(processed_files)}ファイル")
    print(f"未処理: {total_remaining}ファイル")
    print(f"現在のバッチ位置: {current_batch}/{total_batches}（0ベース）")
    
    # 未処理ファイルがない場合は終了
    if total_remaining == 0:
        print("すべてのファイルが既に処理済みです。終了します。")
        return results_file
        
    # バッチ番号の確認と修正（必要に応じて）
    if current_batch >= total_batches:
        print(f"警告: チェックポイントのバッチ位置({current_batch})が合計バッチ数({total_batches})以上です")
        restart_batch = input("バッチ処理を最初から開始しますか？(y/n): ").strip().lower()
        if restart_batch == 'y':
            current_batch = 0
            print("バッチ処理を最初から開始します")
        else:
            print("処理を終了します")
            return results_file
    
    # 軽量パーサーの使用有無
    use_lightweight = USE_LIGHTWEIGHT_PARSER and MIDO_AVAILABLE
    if USE_LIGHTWEIGHT_PARSER and not MIDO_AVAILABLE:
        print("警告: midoパッケージが利用できないため、軽量パーサーは使用できません")
    print(f"軽量パーサーの使用: {use_lightweight}")
    
    # バッチ処理の開始
    total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"合計バッチ数: {total_batches}")
    print(f"バッチ処理を開始します...")
    
    # バッチ番号の確認と修正
    if current_batch > total_batches:
        print(f"警告: チェックポイントのバッチ位置({current_batch})が合計バッチ数({total_batches})を超えています")
        restart_batch = input("バッチ処理を最初から開始しますか？(y/n): ").strip().lower()
        if restart_batch == 'y':
            current_batch = 0
            print("バッチ処理を最初から開始します")
        else:
            print("処理を終了します")
            return results_file
            
    # 既に処理された分をスキップ
    skip_files = processed_count
    if skip_files > 0 and current_batch > 0:
        print(f"既に処理された約{skip_files}ファイルをスキップし、バッチ{current_batch}から再開します")
    
    for i in range(current_batch, total_batches):
        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_remaining)
        current_batch_files = files_to_process[batch_start:batch_end]
        
        print(f"\nバッチ {i+1}/{total_batches} 処理中 ({len(current_batch_files)} ファイル)")
        batch_start_time = time.time()
        
        # バッチ処理を実行（タイムアウト機構付き）
        try:
            print(f"バッチ処理開始: {len(current_batch_files)}ファイル...")
            results = process_batch_with_pool(current_batch_files, use_lightweight)
            
            if results:
                # 処理済みファイルを追跡
                for res in results:
                    if res and 'filename' in res:
                        processed_files.add(res['filename'])
                
                # 結果を一括でCSVに書き込む
                write_results_to_csv(results, results_file, 'a')
                print(f"バッチ処理完了: {len(results)}ファイルを処理しました")
            else:
                print("警告: このバッチでは結果が得られませんでした")
        except Exception as e:
            print(f"バッチ処理エラー: {e}")
            print("次のバッチに進みます...")
        
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
    
    # 処理完了後に統計情報を表示（より効率的な方法）
    try:
        # ファイル行数をカウント（wc -l相当の処理）
        def count_lines(filename):
            count = 0
            with open(filename, 'rb') as f:
                # バイナリモードで読み込み、改行文字をカウント
                for _ in f:
                    count += 1
            return count - 1  # ヘッダー行を除く
        
        # 処理済みファイル数をカウント
        file_count = count_lines(results_file)
        
        # 処理時間の統計（サンプリング）
        sample_size = min(1000, file_count)  # 最大1000サンプル
        if sample_size > 0:
            with open(results_file, 'r', encoding='utf-8') as f:
                next(f)  # ヘッダーをスキップ
                lines = []
                
                # サンプリング間隔を計算
                step = max(1, file_count // sample_size)
                
                # サンプルを収集
                line_number = 0
                for line in f:
                    line_number += 1
                    if line_number % step == 0:
                        lines.append(line)
                    if len(lines) >= sample_size:
                        break
                
                # 処理時間を抽出
                total_time = 0
                valid_samples = 0
                
                for line in lines:
                    try:
                        # 最後のカラム（処理時間）を抽出
                        time_str = line.rsplit(',', 1)[-1].strip()
                        proc_time = float(time_str)
                        total_time += proc_time
                        valid_samples += 1
                    except (ValueError, IndexError):
                        continue
                
                if valid_samples > 0:
                    avg_time = total_time / valid_samples
                    print(f"\n{file_count}ファイルからコード進行を抽出しました")
                    print(f"平均処理時間: {avg_time:.2f}秒/ファイル（{valid_samples}サンプルに基づく）")
    except Exception as e:
        print(f"統計情報の生成エラー: {e}")
        print("統計情報なしで続行します")
    
    return results_file

def main():
    """メイン処理関数"""
    try:
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
        
        print(f"最適化されたMIDIコード進行抽出ツール")
        print(f"===================================")
        print(f"入力ディレクトリ: {input_dir}")
        print(f"出力ディレクトリ: {output_dir}")
        print(f"使用ワーカープロセス数: {MAX_WORKERS}")
        print(f"バッチサイズ: {BATCH_SIZE} ファイル")
        print(f"軽量パーサー設定: {USE_LIGHTWEIGHT_PARSER}")
        
        # music21環境の最適化
        print("music21環境の最適化中...")
        optimize_music21()
        
        # テストモードフラグとテスト件数
        test_mode = False  # テストモードの場合はTrue
        test_file_count = 1000  # テストするファイル数
        
        # システム情報表示
        print(f"Python バージョン: {sys.version}")
        print(f"プロセッサコア数: {mp.cpu_count()}")
        print(f"CSV フィールドサイズ制限: {csv.field_size_limit()}")
        
        # 処理開始時間
        start_all = time.time()
        
        # ファイルリストの取得
        file_paths = get_file_list(input_dir)
        if not file_paths:
            print("MIDIファイルが見つかりませんでした。入力ディレクトリを確認してください。")
            return
        
        # テストモードの場合は処理するファイル数を制限
        if test_mode:
            original_count = len(file_paths)
            file_paths = file_paths[:min(test_file_count, original_count)]
            print(f"テストモード: {original_count}ファイル中{len(file_paths)}ファイルのみ処理します")
        
        # バッチ処理実行
        output_file = process_files_in_batches(file_paths, output_dir, checkpoint_file)
        
        # 総処理時間
        total_elapsed = time.time() - start_all
        print(f"\n総処理時間: {total_elapsed:.2f}秒")
        if len(file_paths) > 0:
            print(f"ファイルあたりの平均時間: {total_elapsed/len(file_paths):.2f}秒")
        print(f"\n出力ファイル: {output_file}")
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。途中経過は保存されています。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Windows環境でのマルチプロセッシング対応
    mp.freeze_support()
    main()