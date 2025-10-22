import re
from collections import defaultdict
import pickle
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
def load_conll_data(filepath):
    """
    Đọc tệp dữ liệu định dạng CoNLL.
    Mỗi dòng: 'word tag'
    Các câu cách nhau bằng dòng trống.
    """
    sentences = []
    labels = []
    
    current_sentence = []
    current_labels = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Nếu là dòng trống -> kết thúc câu
                if not line:
                    if current_sentence: # Đảm bảo câu không rỗng
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        
                        # Reset cho câu mới
                        current_sentence = []
                        current_labels = []
                else:
                    # Tách từ và nhãn (xử lý cả tab và dấu cách)
                    parts = line.split() 
                    
                    if len(parts) >= 2:
                        word = parts[0]
                        tag = parts[1]
                        current_sentence.append(word)
                        current_labels.append(tag)
                    else:
                        print(f"Bỏ qua dòng định dạng không hợp lệ: {line}")
        
        # Thêm câu cuối cùng nếu tệp không kết thúc bằng dòng trống
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
            
        print(f"Đã tải thành công {len(sentences)} câu từ {filepath}")
        return sentences, labels
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {filepath}")
        return [], []
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc {filepath}: {e}")
        return [], []
    
class CRFPreprocessor:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.max_len = 0
        
    def clean_text(self, text):
        """Làm sạch văn bản tiếng Việt"""
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)]', '', text) # Xoá ký tự không thuộc bảng chữ cái, số, dấu câu.
        text = re.sub(r'\d+', '<NUM>', text)
        return text
    
    def build_vocab(self, sentences):
        """Xây dựng từ điển từ vựng"""
        word_counts = defaultdict(int)
        
        for sentence in sentences:
            for word in sentence:
                # Không cần clean_text ở đây, vì extract_features sẽ
                # dùng từ gốc, còn sentences_to_indices sẽ clean
                word_counts[word] += 1
        
        # Thêm token đặc biệt (mặc dù CRF không dùng nhiều)
        special_tokens = ['<PAD>', '<UNK>']
        # Lọc các từ xuất hiện ít (có thể điều chỉnh số 2)
        vocab = special_tokens + [word for word, count in word_counts.items() if count >= 2]
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Kích thước từ vựng (>= 2 lần xuất hiện): {len(vocab)}")
        return vocab
    
    def build_label_vocab(self, all_labels):
        """Xây dựng từ điển nhãn"""
        unique_labels = sorted(set([label for labels in all_labels for label in labels]))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        
        print(f"Số lượng nhãn: {len(unique_labels)}")
        print(f"Các nhãn: {unique_labels}")
        return unique_labels
    
    def extract_features(self, sentence):
        """Trích xuất đặc trưng cho từng từ"""
        features = []
        for i, word in enumerate(sentence):
            if not word:
                word = "<EMPTY>" # Xử lý từ rỗng (nếu có)
            
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
            
            # Context features (từ trước/sau)
            if i > 0:
                word_features['prev_word'] = sentence[i-1].lower()
            else:
                word_features['prev_word'] = '<START>' # Thêm token đầu câu
                
            if i < len(sentence) - 1:
                word_features['next_word'] = sentence[i+1].lower()
            else:
                word_features['next_word'] = '<END>' # Thêm token cuối câu
                
            features.append(word_features)
        
        return features
    
    def sentences_to_indices(self, sentences):
        """Chuyển câu thành indices (Dùng cho BiLSTM-CRF, CRF truyền thống không cần)"""
        indices_sentences = []
        for sentence in sentences:
            indices = []
            for word in sentence:
                # Dùng từ gốc, không clean, để tra cứu
                idx = self.word2idx.get(word, self.word2idx['<UNK>'])
                indices.append(idx)
            indices_sentences.append(indices)
        return indices_sentences
    
    def labels_to_indices(self, labels):
        """Chuyển nhãn thành indices (Dùng cho BiLSTM-CRF)"""
        indices_labels = []
        for label_seq in labels:
            indices = [self.label2idx.get(label, 0) for label in label_seq] # Giả sử 0 là nhãn 'O'
            indices_labels.append(indices)
        return indices_labels
    
    def pad_sequences(self, sequences, max_len=None, padding_value=0, padding="post"):
        """Padding sequences về cùng độ dài (Dùng cho BiLSTM-CRF)"""
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) >= max_len:
                padded.append(seq[:max_len])
            else:
                pad_len = max_len - len(seq)
                if padding == "post":
                    padded.append(seq + [padding_value] * pad_len)
                else:
                    padded.append([padding_value] * pad_len + seq)
        return np.array(padded)
    
    def preprocess_data(self, sentences, labels=None):
        """
        Tiền xử lý toàn bộ dữ liệu.
        Xây dựng vocab nếu cần, và luôn trích xuất features.
        """
        # Xây dựng vocabulary nếu chưa có (chỉ làm khi xử lý tập train)
        if not self.word2idx:
            self.build_vocab(sentences)
        if labels and not self.label2idx:
            self.build_label_vocab(labels)
        
        # Tính max_len nếu chưa có
        if not self.max_len:
            self.max_len = max(len(sent) for sent in sentences)
            print(f"Max sequence length (từ tập train): {self.max_len}")
        
        # **QUAN TRỌNG**: Trích xuất features cho CRF
        X_features = [self.extract_features(sent) for sent in sentences]
        
        result = {
            'X_features': X_features,
            'sentences': sentences
        }
        
        # Các phần 'indices' và 'padded' này không cần thiết
        # cho sklearn-crfsuite, nhưng vẫn giữ nếu bạn muốn dùng
        # cho mô hình BiLSTM-CRF sau này.
        X_indices = self.sentences_to_indices(sentences)
        X_padded = self.pad_sequences(X_indices, self.max_len)
        result['X_indices'] = X_padded
        
        if labels:
            result['labels'] = labels # CRF dùng trực tiếp labels (list[list[str]])
            y_indices = self.labels_to_indices(labels)
            y_padded = self.pad_sequences(y_indices, self.max_len, padding_value=-1)
            result['y_indices'] = y_padded
        
        return result
    
    def save_preprocessor(self, filepath):
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
    
    def load_preprocessor(self, filepath):
        """Tải preprocessor"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.label2idx = data['label2idx']
        self.idx2label = data['idx2label']
        self.max_len = data['max_len']
        print(f"Preprocessor đã được tải từ {filepath}")


#################################################################
#  PHẦN 3: KỊCH BẢN HUẤN LUYỆN VÀ ĐÁNH GIÁ
#################################################################

if __name__ == "__main__":
    
    # --- 1. TẢI TẤT CẢ DỮ LIỆU ---
    print("--- 1. Bắt đầu tải dữ liệu ---")
    train_file = r'D:\Project\NLP\data\data_ner\train.conll'
    dev_file = r'D:\Project\NLP\data\data_ner\dev.conll'
    test_file = r'D:\Project\NLP\data\data_ner\test.conll'
    
    X_train_sents, y_train_labels = load_conll_data(train_file)
    X_dev_sents, y_dev_labels = load_conll_data(dev_file)
    X_test_sents, y_test_labels = load_conll_data(test_file)
    
    if not X_train_sents or not X_dev_sents or not X_test_sents:
        print("Lỗi: Không tải được dữ liệu. Vui lòng kiểm tra lại đường dẫn tệp.")
        exit()
        
    print("--- Tải dữ liệu hoàn tất ---")

    # --- 2. TIỀN XỬ LÝ ---
    print("\n--- 2. Bắt đầu tiền xử lý ---")
    preprocessor = CRFPreprocessor()

    # Bước 2a: HỌC từ vựng và trích xuất đặc trưng cho tập TRAIN
    print("Đang xử lý tập Train (Xây dựng vocab)...")
    train_data = preprocessor.preprocess_data(X_train_sents, y_train_labels)
    X_train_features = train_data['X_features']

    # Bước 2b: ÁP DỤNG từ vựng đã học cho tập DEV
    print("Đang xử lý tập Dev...")
    dev_data = preprocessor.preprocess_data(X_dev_sents, y_dev_labels)
    X_dev_features = dev_data['X_features']

    # Bước 2c: ÁP DỤNG từ vựng đã học cho tập TEST
    print("Đang xử lý tập Test...")
    test_data = preprocessor.preprocess_data(X_test_sents, y_test_labels)
    X_test_features = test_data['X_features']
    print("--- Tiền xử lý hoàn tất ---")

    # (Lưu preprocessor để dùng sau này)
    preprocessor.save_preprocessor(r"D:\Project\NLP\result\crf_preprocessor.pkl")


    # --- 3. HUẤN LUYỆN MÔ HÌNH ---
    print("\n--- 3. Bắt đầu huấn luyện CRF ---")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,       # Tham số regularization L1
        c2=0.1,       # Tham số regularization L2
        max_iterations=100,
        all_possible_transitions=True # Cho phép mô hình học các chuyển tiếp nhãn
    )

    # Huấn luyện mô hình
    print("Đang huấn luyện... (có thể mất vài phút)")
    crf.fit(X_train_features, y_train_labels)
    print("--- Huấn luyện hoàn tất ---")

    # (Lưu model đã huấn luyện)
    model_path = r'D:\Project\NLP\result\crf_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(crf, f)
    print(f"Mô hình đã được lưu tại {model_path}")


    # --- 4. ĐÁNH GIÁ ---
    print("\n--- 4. Bắt đầu đánh giá ---")

    # Lấy danh sách nhãn để báo cáo (loại bỏ 'O' để dễ đọc)
    report_labels = [lbl for lbl in preprocessor.label2idx.keys() if lbl != 'O']

    # Đánh giá trên tập DEV (Validation)
    print("====================================")
    print("   Kết quả trên tập DEV (Validation)  ")
    print("====================================")
    y_dev_pred = crf.predict(X_dev_features)
    print(metrics.flat_classification_report(
        y_dev_labels, y_dev_pred, labels=report_labels, digits=4
    ))

    # Đánh giá cuối cùng trên tập TEST
    print("\n====================================")
    print("     Kết quả trên tập TEST (Final)    ")
    print("====================================")
    y_test_pred = crf.predict(X_test_features)
    print(metrics.flat_classification_report(
        y_test_labels, y_test_pred, labels=report_labels, digits=4
    ))