import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Any, Dict
import pickle
import re
import os
from dotenv import load_dotenv
# Added imports for transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json # For loading label mappings

load_dotenv()

# --- ViHealthBERT Model Loading ---
print("Đang khởi tạo và tải model ViHealthBERT")
# *** Adjust this path if your model is saved elsewhere ***
VIHEALTHBERT_MODEL_PATH = os.getenv("VIHEALTHBERT_MODEL_PATH", r"D:\Project\NLP\result\vihealth") # Using the path from the image
print(f"Đường dẫn model ViHealthBERT: {VIHEALTHBERT_MODEL_PATH}")

try:
    vihealth_tokenizer = AutoTokenizer.from_pretrained(VIHEALTHBERT_MODEL_PATH)
    vihealth_model = AutoModelForTokenClassification.from_pretrained(VIHEALTHBERT_MODEL_PATH)
    # Load label mappings saved during ViHealthBERT training (assuming it's in config.json from image)
    with open(os.path.join(VIHEALTHBERT_MODEL_PATH, 'config.json'), 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        # Check if label mappings exist directly or nested
        if 'id2label' in config_data:
             id2label = {int(k): v for k, v in config_data['id2label'].items()} # Ensure keys are integers
             label2id = config_data['label2id']
        elif 'label_mappings' in config_data and 'id2label' in config_data['label_mappings']: # Check nested structure
             id2label = {int(k): v for k, v in config_data['label_mappings']['id2label'].items()}
             label2id = config_data['label_mappings']['label2id']
        else:
             print("!!! LỖI: Không tìm thấy 'id2label' và 'label2id' trong config.json.")
             id2label = None
             label2id = None


    # Move model to appropriate device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vihealth_model.to(device)
    vihealth_model.eval() # Set model to evaluation mode
    print("--- Tải model ViHealthBERT thành công! ---")
except Exception as e:
    print(f"!!! LỖI: Không thể tải model ViHealthBERT từ '{VIHEALTHBERT_MODEL_PATH}'. Lỗi: {e}")
    vihealth_tokenizer = None
    vihealth_model = None
    id2label = None
    label2id = None
    device = None
# --- End ViHealthBERT Loading ---


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
    # (extract_features function remains unchanged)
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
    description="API cho mô hình NER (ViHealthBERT VÀ CRF)",
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

# --- Renamed and Updated API ENDPOINT FOR ViHealthBERT ---
@app.post("/analyze-vihealthbert")
async def analyze_vihealthbert_endpoint(request: TextInput):
    if not vihealth_model or not vihealth_tokenizer or not id2label:
        return {"error": "Model ViHealthBERT chưa được tải."}
    try:
        # --- Start: Alignment logic based on ViHealthBERT.ipynb ---
        text = request.text
        words = text.split()
        if not words:
            return []

        tokens = []
        label_ids_aligned = []
        word_start_indices = [] # Keep track of the token index for the start of each word

        # Add CLS token and placeholder label
        tokens.append(vihealth_tokenizer.cls_token)
        label_ids_aligned.append(-100) # Ignore CLS for prediction mapping

        # Tokenize word by word and align labels
        max_len = vihealth_tokenizer.model_max_length if hasattr(vihealth_tokenizer, 'model_max_length') else 256 # Use model's max_len or default
        current_length = 1 # Start with CLS

        for word_idx, word in enumerate(words):
            word_tokens = vihealth_tokenizer.tokenize(word)
            if not word_tokens: # Handle empty tokens if tokenizer produces them
                 continue

            # Check for truncation before adding word tokens
            if current_length + len(word_tokens) >= max_len -1: # Need space for SEP
                print(f"Warning: Input truncated at word '{word}' due to max length {max_len}")
                break # Stop adding words if max length is exceeded

            word_start_indices.append(len(tokens)) # Store the index of the first token of this word
            tokens.extend(word_tokens)
            current_length += len(word_tokens)

            # Assign label -100 to all tokens (we'll use word_start_indices later)
            # This simplifies inference as we only need the prediction for the first token
            label_ids_aligned.extend([-100] * len(word_tokens))


        # Add SEP token and placeholder label
        tokens.append(vihealth_tokenizer.sep_token)
        label_ids_aligned.append(-100)

        # Convert tokens to IDs and create attention mask
        input_ids = vihealth_tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # Padding (if necessary, though less common with manual tokenization like this)
        # padding_length = max_len - len(input_ids)
        # if padding_length > 0:
        #     input_ids = input_ids + ([vihealth_tokenizer.pad_token_id] * padding_length)
        #     attention_mask = attention_mask + ([0] * padding_length)
        #     label_ids_aligned = label_ids_aligned + ([-100] * padding_length)


        # Convert to tensors and move to device
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)
        # labels_tensor = torch.tensor([label_ids_aligned], dtype=torch.long).to(device) # Labels not needed for inference

        # Perform inference
        with torch.no_grad():
            outputs = vihealth_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

        # Get predicted label IDs for all tokens
        predictions = torch.argmax(outputs.logits, dim=2)[0] # Get predictions for the first batch item

        # Map predictions back to original words using word_start_indices
        results = []
        for word_idx, original_word in enumerate(words):
            if word_idx >= len(word_start_indices): # Check if word was truncated
                 break
            start_token_index = word_start_indices[word_idx]
            if start_token_index >= len(predictions): # Safety check for prediction length
                pred_id = label2id.get('O', 0) # Default to 'O' if index out of bounds
            else:
                 pred_id = predictions[start_token_index].item() # Get prediction for the first token of the word

            label = id2label.get(pred_id, 'O') # Map ID to label, default to 'O'
            results.append((original_word, label))

        # --- End: Alignment logic based on ViHealthBERT.ipynb ---

        return results

    except Exception as e:
        print(f"Lỗi khi dự đoán ViHealthBERT: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # Return error as JSON for frontend to handle
        return {"error": f"Lỗi xử lý ViHealthBERT: {str(e)}"}

# --- API ENDPOINT CHO MODEL CRF (unchanged) ---
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
        # Return error as JSON
        return {"error": f"Lỗi xử lý CRF: {str(e)}"}

@app.get("/")
def read_root():
    return {
        "message": "Chào mừng đến với ViMedNER API",
        "endpoints": {
            "/analyze-vihealthbert": "Phân tích với ViHealthBERT",
            "/analyze-crf": "Phân tích với CRF"
        },
        "status": {
            "vihealthbert": "ready" if (vihealth_model and vihealth_tokenizer) else "error",
            "crf": "ready" if (crf_model and crf_preprocessor_data) else "error"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "vihealthbert": vihealth_model is not None and vihealth_tokenizer is not None,
            "crf": crf_model is not None and crf_preprocessor_data is not None
        }
    }

# --- Added main block to run with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # Make sure port matches the one expected by frontend (8001)
    uvicorn.run(app, host="127.0.0.1", port=8001)