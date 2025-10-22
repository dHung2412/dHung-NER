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

# --- 1. TẢI MODEL PhoBERT + Linear ---
print("Đang khởi tạo và tải model PhoBERT + Linear...")
MODEL_PATH = "models/best_model"
try:
    inference_engine = ViMedNERInference(model_path=MODEL_PATH)
    print("--- Tải model PhoBERT + Linear thành công! ---")
except Exception as e:
    print(f"!!! LỖI: Không thể tải model PhoBERT từ '{MODEL_PATH}'. Lỗi: {e}")
    inference_engine = None 

# --- 2. TẢI MODEL CRF (CỔ ĐIỂN) ---
print("Đang khởi tạo và tải model CRF (.pkl)...")
CRF_PREPROCESSOR_PATH = os.getenv("CRF_PREPROCESSOR_PATH")
CRF_MODEL_PATH = os.getenv("CRF_MODEL_PATH")

try:
    # Tải file preprocessor (là một DICT)
    with open(CRF_PREPROCESSOR_PATH, 'rb') as f:
        crf_preprocessor_data = pickle.load(f)
    
    # Tải file model (là model CRF đã train)
    with open(CRF_MODEL_PATH, 'rb') as f:
        crf_model = pickle.load(f)
        
    print(f"--- Tải model CRF (.pkl) từ '{CRF_MODEL_PATH}' thành công! ---")
except Exception as e:
    print(f"!!! LỖI: Không thể tải 2 file CRF (.pkl). Lỗi: {e}")
    crf_preprocessor_data = None
    crf_model = None

# --- 3. SAO CHÉP HÀM EXTRACT_FEATURES TỪ CODE CỦA BẠN ---
# Đây chính là hàm trích xuất đặc trưng mà server đang thiếu
def extract_features(sentence: List[str]) -> List[Dict[str, Any]]:
    """Trích xuất đặc trưng cho từng từ (giống lúc train)."""
    features = []
    for i, word in enumerate(sentence):
        word_features = {
            'word': word.lower(),
            'is_first': i == 0,
            'is_last': i == len(sentence) - 1,
            'is_capitalized': word[0].upper() == word[0],
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
            # (Bạn có thể thêm các feature khác từ file code của bạn nếu cần)
            'prev_word': sentence[i - 1].lower() if i > 0 else '<START>',
            'next_word': sentence[i + 1].lower() if i < len(sentence) - 1 else '<END>',
        }
        features.append(word_features)
    return features

# --- 4. KHỞI TẠO API SERVER ---
app = FastAPI(
    title="ViMedNER API",
    description="API cho mô hình NER (PhoBERT + Linear VÀ CRF Cổ điển)",
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

# --- 5. TẠO API ENDPOINT CHO PhoBERT + Linear ---
@app.post("/analyze-linear", response_model=List[Tuple[str, str]])
async def analyze_linear_endpoint(request: TextInput):
    if not inference_engine:
        return {"error": "Model PhoBERT + Linear chưa được tải."}, 500
    try:
        predictions = inference_engine.predict_sentence(request.text)
        return predictions 
    except Exception as e:
        return {"error": str(e)}, 500

# --- 6. TẠO ENDPOINT MỚI CHO MODEL CRF (.pkl) ---
@app.post("/analyze-crf", response_model=List[Tuple[str, str]])
async def analyze_crf_endpoint(request: TextInput):
    if not crf_preprocessor_data or not crf_model:
        return {"error": "Model CRF (.pkl) chưa được tải."}, 500

    try:
        # 1. Tách câu thành các từ
        words = request.text.split()
        if not words:
            return []

        # 2. Áp dụng HÀM trích xuất đặc trưng (ĐÃ SỬA)
        features = extract_features(words)
        
        # 3. Dự đoán bằng model CRF
        # Gói 'features' vào 1 list vì model mong đợi 1 list các câu
        labels = crf_model.predict([features])
        
        # 4. Lấy kết quả
        predicted_labels = labels[0] # Lấy list nhãn của câu đầu tiên

        # 5. Gộp từ và nhãn lại
        return list(zip(words, predicted_labels))

    except Exception as e:
        print(f"Lỗi khi dự đoán CRF: {e}") 
        return {"error": str(e)}, 500

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với ViMedNER API. Các endpoints: /analyze-linear, /analyze-crf"}