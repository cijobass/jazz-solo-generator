# data_preparation.py
import os
import pandas as pd
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, normalizers

# 数字配列をコード名に変換するためのマッピングテーブル
PITCH_TO_NOTE = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'
}

# コードタイプのマッピング（簡略化版）
# 実際には、より複雑なコード構造の識別が必要
CHORD_TYPE_MAP = {
    '0-4-7': 'maj',      # 長三和音
    '0-3-7': 'min',      # 短三和音
    '0-4-7-10': '7',     # ドミナント7th
    '0-4-7-11': 'maj7',  # メジャー7th
    '0-3-7-10': 'm7',    # マイナー7th
    '0-3-6-10': 'm7b5',  # ハーフディミニッシュ
    '0-3-6-9': 'dim7',   # ディミニッシュ7th
}

def convert_chord_numbers_to_name(chord_numbers):
    """
    数字表記のコードを実際のコード名に変換する関数
    
    Args:
        chord_numbers: 文字列（例: "0-4-7"）または数字のリスト
    
    Returns:
        str: コード名（例: "Cmaj"）
    """
    if isinstance(chord_numbers, str):
        # ハイフンで分割して数字のリストに変換
        notes = [int(n) for n in chord_numbers.split('-') if n.isdigit()]
    else:
        notes = chord_numbers
    
    if not notes:
        return "N.C."  # No Chord
    
    # ルート音を特定
    root = notes[0] % 12
    root_name = PITCH_TO_NOTE[root]
    
    # ルート音からの相対位置に変換
    relative_notes = [(n - root) % 12 for n in notes]
    relative_notes = sorted(list(set(relative_notes)))  # 重複を削除して昇順ソート
    
    # コード構造の文字列表現（例: "0-4-7"）
    chord_structure = '-'.join([str(n) for n in relative_notes])
    
    # マッピングテーブルからコードタイプを取得
    chord_type = CHORD_TYPE_MAP.get(chord_structure, '')
    
    # コード名を構築
    if chord_type:
        return f"{root_name}{chord_type}"
    else:
        # 未知のコード構造の場合は数字表記のまま返す
        return f"{root_name}({'-'.join([str(n) for n in notes])})"

class ChordProgressionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, convert_to_chord_names=False):
        """
        コード進行データセットの初期化
        
        Args:
            file_path: CSVファイルのパス
            tokenizer: コード進行をトークン化するトークナイザー
            max_length: シーケンスの最大長
            convert_to_chord_names: コードの数字表記を実際のコード名に変換するかどうか
        """
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.convert_to_chord_names = convert_to_chord_names
        
        # コード進行の前処理
        self.chord_progressions = []
        chord_data = self.data['chords'].astype(str).tolist()
        
        for progression in chord_data:
            # パイプ記号による分割
            if '|' in progression:
                # 小節単位のコード進行を抽出
                chords = progression.split('|')
                
                # Chord-接頭辞を削除
                cleaned_chords = []
                for chord in chords[:30]:  # 最初の30小節だけを使用
                    # Chord-接頭辞を削除し、数字のみを抽出
                    if chord.startswith('Chord-'):
                        chord = chord.replace('Chord-', '')
                    
                    # ハイフンで区切られた数字の処理
                    # 長すぎるコード表現を避けるために最初の4つの数字だけを使用
                    parts = chord.split('-')
                    if len(parts) > 4:
                        parts = parts[:4]
                    
                    # 数字以外の文字を削除
                    parts = [p for p in parts if p.isdigit()]
                    
                    if parts:
                        # コード名に変換する場合
                        if self.convert_to_chord_names:
                            try:
                                chord_name = convert_chord_numbers_to_name(parts)
                                cleaned_chords.append(chord_name)
                            except:
                                # 変換エラーが発生した場合は元の数字表記を使用
                                cleaned_chords.append('-'.join(parts))
                        else:
                            # 数字表記のまま使用
                            cleaned_chords.append('-'.join(parts))
                
                if cleaned_chords:
                    # 小節単位での区切りを保持するためにコンマで結合
                    self.chord_progressions.append(','.join(cleaned_chords))
        
        print(f"有効なコード進行数: {len(self.chord_progressions)}")
        if len(self.chord_progressions) > 0:
            print(f"サンプル: {self.chord_progressions[0][:100]}...")
            
            # コード進行のマッピングを保存（後で生成時に使用）
            self.save_chord_mapping()
    
    def save_chord_mapping(self):
        """コード進行のマッピングを保存"""
        # 全コード進行から一意なコードパターンを抽出
        unique_chords = set()
        for progression in self.chord_progressions:
            chords = progression.split(',')
            unique_chords.update(chords)
        
        # コード名変換の辞書を作成
        chord_mapping = {}
        
        for chord in unique_chords:
            # 数字表記のコードのみを処理
            if '-' in chord and all(p.isdigit() for p in chord.split('-')):
                try:
                    chord_name = convert_chord_numbers_to_name(chord)
                    chord_mapping[chord] = chord_name
                except:
                    pass
        
        # マッピングをJSON形式で保存
        with open('models/chord_mapping.json', 'w') as f:
            json.dump(chord_mapping, f, indent=2)
        
        print(f"コードマッピングを保存しました。一意なコード数: {len(chord_mapping)}")
    
    def __len__(self):
        return len(self.chord_progressions)
    
    def __getitem__(self, idx):
        chord_progression = self.chord_progressions[idx]
        
        # トークン化
        inputs = self.tokenizer(
            chord_progression,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 入力と同じシーケンスを出力としても使用（自己回帰モデル用）
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        
        # シフトした形でターゲットを作成（次のトークンを予測）
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.pad_token_id
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def create_chord_tokenizer(chord_progressions, vocab_size=10000, save_path="models/tokenizer"):
    """
    コード進行専用のトークナイザーを作成
    
    Args:
        chord_progressions: コード進行のリスト
        vocab_size: 語彙サイズ
        save_path: トークナイザーを保存するパス
    
    Returns:
        PreTrainedTokenizerFast: 学習済みトークナイザー
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # ユニークなコード進行パターンを分析
    all_tokens = []
    for progression in chord_progressions:
        tokens = progression.split(',')
        all_tokens.extend(tokens)
    
    unique_tokens = set(all_tokens)
    print(f"ユニークなコードパターン数: {len(unique_tokens)}")
    print(f"サンプル: {list(unique_tokens)[:10]}")
    
    # WordPieceモデルのトークナイザーを作成
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    
    # ノーマライザーの設定
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(r"\s+", " "),
        normalizers.Strip()
    ])
    
    # プリトークナイザーの設定（カンマと数字の組み合わせを適切に扱う）
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=",", behavior="isolated"),
        pre_tokenizers.WhitespaceSplit()
    ])
    
    # トレーナーの設定
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        continuing_subword_prefix="##"
    )
    
    # トークナイザーの訓練
    tokenizer.train_from_iterator(chord_progressions, trainer)
    
    # 後処理の設定
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]"))
        ]
    )
    
    # トークナイザーを保存
    tokenizer.save(f"{save_path}/tokenizer.json")
    
    # Transformers互換のトークナイザーに変換
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{save_path}/tokenizer.json",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    return fast_tokenizer

def prepare_chord_data(csv_path, batch_size=32, train_ratio=0.8, convert_to_chord_names=False):
    """
    コード進行データの準備
    
    Args:
        csv_path: CSVファイルのパス
        batch_size: バッチサイズ
        train_ratio: 訓練データの割合
        convert_to_chord_names: コードの数字表記を実際のコード名に変換するかどうか
    
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    # トークナイザーの作成または読み込み
    tokenizer_path = "models/tokenizer"
    if os.path.exists(f"{tokenizer_path}/tokenizer.json"):
        print(f"既存のトークナイザーを読み込みます: {tokenizer_path}/tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{tokenizer_path}/tokenizer.json",
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )
    else:
        print("新しいトークナイザーを作成します")
        # データセットを作成して前処理されたコード進行を取得
        temp_dataset = ChordProgressionDataset(csv_path, None, convert_to_chord_names=convert_to_chord_names)
        tokenizer = create_chord_tokenizer(temp_dataset.chord_progressions, save_path=tokenizer_path)
    
    # データセットの作成（トークナイザーを渡して）
    dataset = ChordProgressionDataset(csv_path, tokenizer, convert_to_chord_names=convert_to_chord_names)
    
    # 訓練データと検証データに分割
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, tokenizer