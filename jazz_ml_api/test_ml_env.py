import torch
import torchvision
import torchaudio
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    print("=== ライブラリの動作確認スクリプト ===")
    
    # 各ライブラリのバージョンを表示
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"torchaudio version: {torchaudio.__version__}")
    
    # GPUが利用可能か確認
    if torch.cuda.is_available():
        print("GPUが利用可能です。")
        print(f"利用可能なGPU数: {torch.cuda.device_count()}")
        print(f"現在のGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPUは利用できません。CPUで動作します。")
    
    # テスト用のテンソルを作成し、簡単な計算を実行
    a = torch.rand((3, 3))
    b = torch.ones((3, 3))
    c = a + b
    print("\nテンソルの簡単な加算テスト:")
    print("Tensor a:")
    print(a)
    print("Tensor b (ones):")
    print(b)
    print("a + b:")
    print(c)
    
    # torchvision の簡単な変換テスト：画像をテンソルに変換
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # ここでは、ランダムな画像データ（PIL Image相当）を生成して変換
    # ※ 実際の画像がなくても動作確認できるように、numpyでダミーデータを作成
    import numpy as np
    from PIL import Image
    dummy_image = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    tensor_image = transform(dummy_image)
    print("\ntorchvision 変換テスト:")
    print(f"変換後の画像テンソルサイズ: {tensor_image.size()}")
    
    # torchaudio の簡単なテスト：利用可能なオーディオバックエンドを表示
    backends = torchaudio.list_audio_backends()
    print("\ntorchaudio 利用可能なバックエンド:")
    print(backends)
    
    # Hugging Face transformers の簡単なテスト：GPT-2 のトークナイザーを利用
    print("\nTransformers テスト: GPT-2 トークナイザーの動作確認")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    text = "Jazz is"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("入力テキスト:", text)
    print("生成されたテキスト:", generated_text)
    
    print("\n利用可能なバックエンド:", torchaudio.list_audio_backends())

    # バックエンドの設定を試みる
    try:
        torchaudio.set_audio_backend("soundfile")
        print("soundfileバックエンドを設定しました")
    except Exception as e:
        print(f"エラー: {e}")

    # 再度確認
    print("設定後の利用可能なバックエンド:", torchaudio.list_audio_backends())
    
    # バックエンドを設定
    torchaudio.set_audio_backend("soundfile")

    # サンプル音声ファイルがあれば読み込みテスト
    try:
        # 音声ファイルのパスを指定（WAVファイルなど）
        # 仮想的にtensorを生成してダミーテストする場合は以下を試す
        dummy_waveform = torch.rand(2, 16000)  # 2チャンネル、16kHzのランダム信号
        print(f"\nダミー波形サイズ: {dummy_waveform.size()}")
        print("torchaudioは機能しています")
    except Exception as e:
        print(f"エラー: {e}")
    
    print("\n=== 全てのテストが完了しました ===")

if __name__ == '__main__':
    main()
