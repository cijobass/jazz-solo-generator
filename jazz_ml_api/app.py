# app.py
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
from transformers import PreTrainedTokenizerFast
from model import ChordProgressionModel

# アプリケーションの初期化
app = FastAPI(
    title="Jazz Chord Progression API",
    description="ジャズコード進行生成API",
    version="0.1.0"
)

# モデルとトークナイザー
model = None
tokenizer = None

# モデルのロード関数
def load_model():
    global model, tokenizer
    
    try:
        # トークナイザーのロード
        tokenizer_path = "models/tokenizer/tokenizer.json"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>"
        )
        
        # モデルのロード
        model_path = "models/chord_model/best_model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChordProgressionModel(
            vocab_size=len(tokenizer),
            device=device
        )
        model.load(model_path)
        
        print(f"Model loaded successfully. Using device: {device}")
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# リクエスト/レスポンスモデル
class GenerationRequest(BaseModel):
    prompt: Optional[str] = None
    max_length: Optional[int] = 128
    num_sequences: Optional[int] = 1
    temperature: Optional[float] = 1.0

class GenerationResponse(BaseModel):
    chord_progressions: List[str]
    prompt: Optional[str]

# 起動時にモデルをロード
@app.on_event("startup")
async def startup_event():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model)

# ヘルスチェックエンドポイント
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        return {"status": "loading", "message": "Model is still loading"}
    return {"status": "healthy", "message": "API is running"}

# コード進行生成エンドポイント
@app.post("/generate", response_model=GenerationResponse)
async def generate_chord_progression(request: GenerationRequest):
    if model is None or tokenizer is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    try:
        generated_progressions = model.generate(
            tokenizer,
            prompt=request.prompt,
            max_length=request.max_length,
            num_return_sequences=request.num_sequences,
            temperature=request.temperature
        )
        
        return {
            "chord_progressions": generated_progressions,
            "prompt": request.prompt
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# 入力から次のコードを予測するエンドポイント
@app.post("/predict_next")
async def predict_next_chord(request: GenerationRequest):
    if model is None or tokenizer is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    try:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required for prediction")
        
        # 短めの生成（次のコードのみを予測）
        generated = model.generate(
            tokenizer,
            prompt=request.prompt,
            max_length=len(request.prompt.split(',')) + 3,  # 少し余裕を持たせる
            num_return_sequences=1,
            temperature=request.temperature
        )[0]
        
        # 入力されたプロンプトを除去して次のコードのみを取得
        next_chords = generated[len(request.prompt):].strip()
        if next_chords.startswith(','):
            next_chords = next_chords[1:].strip()
        
        return {
            "input": request.prompt,
            "predicted_next": next_chords
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)