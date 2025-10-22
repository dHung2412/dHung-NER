import torch
from transformers import AutoTokenizer
import json
from typing import List, Tuple
from src.web.backend.model import PhoBERTForNER

class ViMedNERInference:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load config
        with open(f"{model_path}/config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load label mappings
        with open(f"{model_path}/label_mappings.json", 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            self.label_to_id = mappings['label_to_id']
            self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = PhoBERTForNER(
            model_name=self.config['model_name'],
            num_labels=self.config['num_labels'],
            dropout=self.config['dropout']
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Available labels: {list(self.label_to_id.keys())}")
    
    def predict_sentence(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Dự đoán nhãn NER cho một câu đơn.
        Đây là hàm duy nhất được 'api_server.py' gọi đến.
        """
        # Tokenize câu (Word-level)
        words = sentence.split()
        
        tokens = []
        word_to_token_map = []
        
        # Thêm [CLS] token
        tokens.append(self.tokenizer.cls_token)
        
        for word in words:
            word_start_idx = len(tokens)
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            word_to_token_map.append(word_start_idx) # Lưu vị trí của token đầu tiên của mỗi từ
        
        # Thêm [SEP] token
        tokens.append(self.tokenizer.sep_token)
        
        # Cắt ngắn nếu dài hơn max_length
        max_len = self.config.get('max_length', 128) # Lấy max_length từ config, mặc định 128
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # Chuyển tokens sang IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Padding
        while len(input_ids) < max_len:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
        
        # Chuyển sang Tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # Lấy dự đoán (đã tối ưu)
        with torch.no_grad():
            # Gọi hàm .predict() của model thay vì .forward()
            predictions = self.model.predict(input_ids, attention_mask)
        
        # Map kết quả dự đoán (IDs) về lại các từ (words)
        result = []
        for i, word in enumerate(words):
            label = 'O' # Mặc định là 'O'
            if i < len(word_to_token_map):
                token_idx = word_to_token_map[i]
                if token_idx < predictions.size(1):
                    pred_id = predictions[0][token_idx].item()
                    label = self.id_to_label.get(pred_id, 'O') # Dùng .get() để tránh lỗi
            
            result.append((word, label))
        
        return result
