# data_preparation.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

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
        
        # コード進行のみを抽出（不要なカラムを削除）
        self.chord_progressions = self.data['chords'].tolist()
        
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

def create_chord_tokenizer(chord_progressions, vocab_size=5000, save_path="models/tokenizer"):
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
    
    # BPEモデルを使用
    tokenizer = Tokenizer(models.BPE())
    
    # プリトークナイザー設定
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    # トレーナーの設定
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )
    
    # トークナイザーの訓練
    tokenizer.train_from_iterator(chord_progressions, trainer)
    
    # 後処理の設定（EOS, BOSトークンの追加など）
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>"))
        ]
    )
    
    # トークナイザーを保存
    tokenizer.save(f"{save_path}/tokenizer.json")
    
    # Transformers互換のトークナイザーに変換
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{save_path}/tokenizer.json",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>"
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
    # データ読み込み
    data = pd.read_csv(csv_path)
    chord_progressions = data['chords'].tolist()
    
    # トークナイザーの作成または読み込み
    tokenizer_path = "models/tokenizer"
    if os.path.exists(f"{tokenizer_path}/tokenizer.json"):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{tokenizer_path}/tokenizer.json",
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>"
        )
    else:
        tokenizer = create_chord_tokenizer(chord_progressions, save_path=tokenizer_path)
    
    # データセットの作成
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