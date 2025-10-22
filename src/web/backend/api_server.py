import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Any, Dict
import pickle  
import re 
from src.web.backend.inference import ViMedNERInference
import os
from dotenv import load_dotenv
load_dotenv()

print("Đang khởi tạo và tải model PhoBERT")
MODEL_PATH = os.getenv("PHOBERT_MODEL_PATH", "result/phobert/best_models")
print(f"Đường dẫn model PhoBERT: {MODEL_PATH}")

try:
    inference_engine = ViMedNERInference(model_path=MODEL_PATH)
    print("--- Tải model PhoBERT + Linear thành công! ---")
except Exception as e:
    print(f"!!! LỖI: Không thể tải model PhoBERT từ '{MODEL_PATH}'. Lỗi: {e}")
    inference_engine = None 

print("Đang khởi tạo và tải model CRF")
CRF_PREPROCESSOR_PATH = os.getenv("CRF_PREPROCESSOR_PATH")
CRF_MODEL_PATH = os.getenv("CRF_MODEL_PATH")
print(f"Đường dẫn CRF preprocessor: {CRF_PREPROCESSOR_PATH}")
print(f"Đường dẫn CRF model: {CRF_MODEL_PATH}")

try:
    with open(CRF_PREPROCESSOR_PATH, 'rb') as f:
        crf_preprocessor_data = pickle.load(f)
    with open(CRF_MODEL_PATH, 'rb') as f:
        crf_model = pickle.load(f)    
    print(f"--- Tải model CRF (.pkl) từ '{CRF_MODEL_PATH}' thành công! ---")
except Exception as e:
    print(f"!!! LỖI: Không thể tải 2 file CRF (.pkl). Lỗi: {e}")
    crf_preprocessor_data = None
    crf_model = None

def extract_features(sentence: List[str]) -> List[Dict[str, Any]]:
    """Trích xuất đặc trưng cho từng từ (giống lúc train)."""
    features = []
    for i, word in enumerate(sentence):
        word_features = {
            'word': word.lower(),
            'is_first': i == 0,
            'is_last': i == len(sentence) - 1,
            'is_capitalized': word[0].upper() == word[0] if word else False,
            'is_all_caps': word.upper() == word,
            'is_all_lower': word.lower() == word,
            'prefix_1': word[:1],
            'prefix_2': word[:2],
            'prefix_3': word[:3],
            'suffix_1': word[-1:],
            'suffix_2': word[-2:],
            'suffix_3': word[-3:],
            'has_digit': bool(re.search(r'\d', word)),
            'has_hyphen': '-' in word,
            'word_length': len(word),
            'prev_word': sentence[i - 1].lower() if i > 0 else '<START>',
            'next_word': sentence[i + 1].lower() if i < len(sentence) - 1 else '<END>',
        }
        features.append(word_features)
    return features

app = FastAPI(
    title="ViMedNER API",
    description="API cho mô hình NER (PhoBERT VÀ CRF)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class TextInput(BaseModel):
    text: str

class ErrorResponse(BaseModel):
    error: str

# --- API ENDPOINT CHO PhoBERT + Linear ---
@app.post("/analyze-linear")
async def analyze_linear_endpoint(request: TextInput):
    if not inference_engine:
        return {"error": "Model PhoBERT + Linear chưa được tải."}
    try:
        predictions = inference_engine.predict_sentence(request.text)
        return predictions 
    except Exception as e:
        print(f"Lỗi khi dự đoán PhoBERT: {e}")
        return {"error": str(e)}

# --- API ENDPOINT CHO MODEL CRF ---
@app.post("/analyze-crf")
async def analyze_crf_endpoint(request: TextInput):
    if not crf_preprocessor_data or not crf_model:
        return {"error": "Model CRF (.pkl) chưa được tải."}

    try:
        # 1. Tách câu thành các từ
        words = request.text.split()
        if not words:
            return []

        # 2. Trích xuất đặc trưng
        features = extract_features(words)
        
        # 3. Dự đoán bằng model CRF
        labels = crf_model.predict([features])
        
        # 4. Lấy kết quả
        predicted_labels = labels[0]

        # 5. Gộp từ và nhãn lại
        return list(zip(words, predicted_labels))

    except Exception as e:
        print(f"Lỗi khi dự đoán CRF: {e}") 
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {
        "message": "Chào mừng đến với ViMedNER API",
        "endpoints": {
            "/analyze-linear": "Phân tích với PhoBERT",
            "/analyze-crf": "Phân tích với CRF"
        },
        "status": {
            "phobert": "ready" if inference_engine else "error",
            "crf": "ready" if (crf_model and crf_preprocessor_data) else "error"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "phobert": inference_engine is not None,
            "crf": crf_model is not None and crf_preprocessor_data is not None
        }
    }