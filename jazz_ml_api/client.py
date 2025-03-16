# client.py
import requests
import json

def generate_chord_progression(prompt=None, num_sequences=3, temperature=0.8):
    """
    APIを使用してコード進行を生成する関数
    
    Args:
        prompt: 開始コード進行（例: "Dm7,G7"）
        num_sequences: 生成する進行の数
        temperature: 生成の多様性（0.0-1.0）
    
    Returns:
        list: 生成されたコード進行のリスト
    """
    url = "http://localhost:8000/generate"
    
    payload = {
        "prompt": prompt,
        "num_sequences": num_sequences,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["chord_progressions"]
    
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return []

def predict_next_chord(current_progression, temperature=0.5):
    """
    現在のコード進行から次のコードを予測する関数
    
    Args:
        current_progression: 現在のコード進行
        temperature: 生成の多様性
    
    Returns:
        str: 予測された次のコード
    """
    url = "http://localhost:8000/predict_next"
    
    payload = {
        "prompt": current_progression,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["predicted_next"]
    
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return ""

if __name__ == "__main__":
    # サンプル実行
    print("=== コード進行生成のテスト ===")
    
    # 2-5-1進行から始める
    prompt = "Dm7,G7,Cmaj7"
    progressions = generate_chord_progression(prompt=prompt, num_sequences=3)
    
    print(f"プロンプト: {prompt}")
    for i, prog in enumerate(progressions):
        print(f"生成 {i+1}: {prog}")
    
    print("\n=== 次のコード予測のテスト ===")
    
    # 現在の進行
    current = "Fmaj7,Em7,Dm7"
    next_chord = predict_next_chord(current)
    
    print(f"現在の進行: {current}")
    print(f"予測された次のコード: {next_chord}")