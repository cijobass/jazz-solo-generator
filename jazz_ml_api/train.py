# train.py
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_preparation import prepare_chord_data
from model import ChordProgressionModel

def train(
    csv_path,
    output_dir="models/chord_model",
    batch_size=32,
    epochs=20,
    learning_rate=5e-5,
    max_length=128,
    n_layer=6,
    n_head=8,
    n_embd=512
):
    """
    コード進行モデルの訓練
    
    Args:
        csv_path: コード進行CSVファイルのパス
        output_dir: モデル出力ディレクトリ
        batch_size: バッチサイズ
        epochs: エポック数
        learning_rate: 学習率
        max_length: 最大シーケンス長
        n_layer: Transformerレイヤー数
        n_head: 注意ヘッド数
        n_embd: 埋め込みサイズ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データの準備
    train_loader, val_loader, tokenizer = prepare_chord_data(
        csv_path,
        batch_size=batch_size
    )
    
    # モデルの初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChordProgressionModel(
        vocab_size=len(tokenizer),
        max_length=max_length,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        device=device
    )
    
    # オプティマイザーの設定
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=learning_rate
    )
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.9
    )
    
    # 訓練ループ
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 訓練
        model.model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            loss = model.train_step(batch, optimizer)
            train_loss += loss
            train_steps += 1
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # 検証
        model.model.eval()
        val_loss = 0
        val_steps = 0
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        for batch in progress_bar:
            loss = model.eval_step(batch)
            val_loss += loss
            val_steps += 1
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 学習率の更新
        scheduler.step()
        
        # モデルの保存（検証ロスが改善した場合）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(os.path.join(output_dir, "best_model"))
            print(f"Model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
            
            # サンプル生成のためのシード（コード進行の数字表現）
            test_prompts = [
                "4-0-4",  # サンプルデータから抽出したコードパターン
                "0-3-6",
                "7-11-2",
                "5-9-0",
                "2-5-10"
            ]
            
            print("Sample generations:")
            for i, prompt in enumerate(test_prompts):
                print(f"\nInput pattern {i+1}: {prompt}")
                samples = model.generate(
                    tokenizer,
                    prompt=prompt,
                    num_return_sequences=1,
                    temperature=0.7,
                    max_length=64
                )
                for j, sample in enumerate(samples):
                    print(f"  Generated: {sample}")
        
        # 最新モデルの保存
        model.save(os.path.join(output_dir, "latest_model"))
    
    # 損失の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a chord progression model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to chord progression CSV file")
    parser.add_argument("--output_dir", type=str, default="models/chord_model", help="Output directory for model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding size")
    
    args = parser.parse_args()
    
    train(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.max_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )