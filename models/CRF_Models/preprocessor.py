import re
import pickle
from collections import defaultdict

class CRFPreprocessor:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.max_len = 0
        
    def _build_vocab(self, sentences):
        """(Internal) Xây dựng từ điển từ vựng"""
        word_counts = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1
        
        special_tokens = ['<PAD>', '<UNK>']
        vocab = special_tokens + [word for word, count in word_counts.items() if count >= 2]
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Kích thước từ vựng (>= 2 lần xuất hiện): {len(vocab)}")

    def _build_label_vocab(self, all_labels):
        """(Internal) Xây dựng từ điển nhãn"""
        unique_labels = sorted(set([label for labels in all_labels for label in labels]))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        print(f"Số lượng nhãn: {len(unique_labels)}")
        print(f"Các nhãn: {unique_labels}")

    def _extract_features(self, sentence):
        """(Internal) Trích xuất đặc trưng cho từng từ trong một câu"""
        features = []
        for i, word in enumerate(sentence):
            if not word:
                word = "<EMPTY>"
            
            word_features = {
                'word': word.lower(),
                'is_first': i == 0,
                'is_last': i == len(sentence) - 1,
                'is_capitalized': word[0].upper() == word[0] if len(word) > 0 else False,
                'is_all_caps': word.upper() == word,
                'is_all_lower': word.lower() == word,
                'prefix_1': word[0] if len(word) > 0 else '',
                'prefix_2': word[:2] if len(word) > 1 else '',
                'prefix_3': word[:3] if len(word) > 2 else '',
                'suffix_1': word[-1] if len(word) > 0 else '',
                'suffix_2': word[-2:] if len(word) > 1 else '',
                'suffix_3': word[-3:] if len(word) > 2 else '',
                'has_digit': bool(re.search(r'\d', word)),
                'has_hyphen': '-' in word,
                'word_length': len(word),
                'is_medical_unit': word.lower() in ['mg', 'ml', 'g', 'kg', 'μg', 'mcg', 'iu', 'mmol'],
            }
            
            if i > 0:
                word_features['prev_word'] = sentence[i-1].lower()
            else:
                word_features['prev_word'] = '<START>'
                
            if i < len(sentence) - 1:
                word_features['next_word'] = sentence[i+1].lower()
            else:
                word_features['next_word'] = '<END>'
                
            features.append(word_features)
        
        return features
    
    def fit_transform(self, sentences, labels):
        """Học vocab và chuyển đổi tập train"""
        print("Đang xử lý tập Train (Xây dựng vocab)...")
        # Xây dựng vocabulary
        if not self.word2idx:
            self._build_vocab(sentences)
        if labels and not self.label2idx:
            self._build_label_vocab(labels)
        
        # Tính max_len
        if not self.max_len:
            self.max_len = max(len(sent) for sent in sentences)
            print(f"Max sequence length (từ tập train): {self.max_len}")
        
        # Trích xuất features
        X_features = [self._extract_features(sent) for sent in sentences]
        return X_features

    def transform(self, sentences):
        """Chỉ chuyển đổi tập dev/test (dùng vocab đã học)"""
        print(f"Đang chuyển đổi {len(sentences)} câu...")
        X_features = [self._extract_features(sent) for sent in sentences]
        return X_features

    def save(self, filepath):
        """Lưu preprocessor"""
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'label2idx': self.label2idx,
            'idx2label': self.idx2label,
            'max_len': self.max_len
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Preprocessor đã được lưu tại {filepath}")

    @staticmethod
    def load(filepath):
        """Tải preprocessor"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = CRFPreprocessor()
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.label2idx = data['label2idx']
        preprocessor.idx2label = data['idx2label']
        preprocessor.max_len = data['max_len']
        print(f"Preprocessor đã được tải từ {filepath}")
        return preprocessor