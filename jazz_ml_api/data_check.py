# data_check.py
import pandas as pd

def check_chord_data(csv_path):
    """コード進行データの内容を確認する関数"""
    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)
    
    # 基本情報の表示
    print(f"データセットの形状: {df.shape}")
    print(f"カラム名: {df.columns.tolist()}")
    
    # 最初の数行のデータ
    print("\n最初の5行:")
    print(df.head())
    
    # chordsカラムの最初の数例を詳しく確認
    if 'chords' in df.columns:
        print("\nコード進行の例:")
        for i in range(min(5, len(df))):
            print(f"\n例 {i+1}:")
            print(f"Raw: {df['chords'].iloc[i]}")
            print(f"Type: {type(df['chords'].iloc[i])}")
            print(f"Length: {len(str(df['chords'].iloc[i]))}")
            
            # 先頭100文字を表示
            chord_str = str(df['chords'].iloc[i])
            print(f"First 100 chars: {chord_str[:100]}...")
            
            # コード進行を分解して表示（カンマ区切りの場合）
            if ',' in chord_str:
                chords = chord_str.split(',')
                print(f"Split by comma - Count: {len(chords)}")
                print(f"First 10 chords: {chords[:10]}")
            
            # パイプ記号で区切られている場合
            if '|' in chord_str:
                measures = chord_str.split('|')
                print(f"Split by pipe - Count: {len(measures)}")
                print(f"First 5 measures: {measures[:5]}")
    
    # 一意なコードの種類を確認（最初の行のみ）
    if 'chords' in df.columns and len(df) > 0:
        first_row = str(df['chords'].iloc[0])
        if ',' in first_row:
            unique_chords = set(first_row.split(','))
            print(f"\n最初の行の一意なコードの種類: {len(unique_chords)}")
            print(f"例: {list(unique_chords)[:10]}")

if __name__ == "__main__":
    # CSVファイルのパスを指定
    csv_path = "../jazz_ai_dataset/processed/chord_progressions_20250316_125653.csv"
    check_chord_data(csv_path)