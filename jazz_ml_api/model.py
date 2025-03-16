# model.py
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class ChordProgressionModel:
    def __init__(
        self,
        vocab_size,
        max_length=128,
        n_layer=6,
        n_head=8,
        n_embd=512,
        device=None
    ):
        """
        コード進行生成モデル
        
        Args:
            vocab_size: トークナイザーの語彙サイズ
            max_length: 最大シーケンス長
            n_layer: Transformerレイヤー数
            n_head: 注意ヘッド数
            n_embd: 埋め込みサイズ
            device: 計算デバイス
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # GPT-2スタイルのモデル設定
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_length,
            n_ctx=max_length,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=vocab_size - 4,  # <bos>トークンID
            eos_token_id=vocab_size - 3,  # <eos>トークンID
            pad_token_id=vocab_size - 1   # <pad>トークンID
        )
        
        # モデルの初期化
        self.model = GPT2LMHeadModel(self.config)
        self.model.to(self.device)
        
    def save(self, path):
        """モデルの保存"""
        self.model.save_pretrained(path)
        
    def load(self, path):
        """モデルの読み込み"""
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.model.to(self.device)
        
    def train_step(self, batch, optimizer):
        """1ステップの訓練"""
        self.model.train()
        
        # データをデバイスに転送
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # モデル出力
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # 勾配の計算と更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def eval_step(self, batch):
        """1ステップの評価"""
        self.model.eval()
        
        with torch.no_grad():
            # データをデバイスに転送
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # モデル出力
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
        return loss.item()
    
    def generate(self, tokenizer, prompt=None, max_length=128, num_return_sequences=1, temperature=1.0):
        """コード進行の生成"""
        self.model.eval()
        
        # プロンプトがない場合は開始トークンから始める
        if prompt is None or prompt == "":
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(self.device)
        else:
            # プロンプトをトークン化
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            )
            input_ids = inputs["input_ids"].to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # トークンをデコード
        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts