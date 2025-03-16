# data_preparation.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, normalizers
import numpy as np

class ChordProgressionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        """
        コード進行データセットの初期化
        
        Args:
            file_path: CSVファイルのパス
            tokenizer: コード進行をトークン化するトークナイザー
            max_length: シーケンスの最大長
        """
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # コード進行のみを抽出
        self.chord_progressions = []
        chord_data = self.data['chords'].astype(str).tolist()
        
        # コード進行の前処理
        for progression in chord_data:
            # パイプ記号による分割
            if '|' in progression:
                # 小節単位のコード進行を抽出
                chords = progression.split('|')
                # Chord-接頭辞を削除し、最初の20小節だけを使用（長すぎるシーケンスを避ける）
                cleaned_chords = [c.replace('Chord-', '') for c in chords[:20]]
                if cleaned_chords:
                    # 小節単位での区切りを保持するためにスペースをカンマに置き換える
                    self.chord_progressions.append(','.join(cleaned_chords))
            else:
                # パイプ記号がない場合はスキップ
                continue
        
        print(f"有効なコード進行数: {len(self.chord_progressions)}")
        if len(self.chord_progressions) > 0:
            print(f"サンプル: {self.chord_progressions[0][:100]}...")
    
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

def prepare_chord_data(csv_path, batch_size=32, train_ratio=0.8):
    """
    コード進行データの準備
    
    Args:
        csv_path: CSVファイルのパス
        batch_size: バッチサイズ
        train_ratio: 訓練データの割合
    
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
        temp_dataset = ChordProgressionDataset(csv_path, None)
        tokenizer = create_chord_tokenizer(temp_dataset.chord_progressions, save_path=tokenizer_path)
    
    # データセットの作成（トークナイザーを渡して）
    dataset = ChordProgressionDataset(csv_path, tokenizer)
    
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