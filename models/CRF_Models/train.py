# file: train.py

from data_loader import load_conll_data
from preprocessor import CRFPreprocessor
from model import CRFModel

# --- 0. ĐỊNH NGHĨA ĐƯỜNG DẪN ---
TRAIN_FILE = r'D:\Project\NLP\data\data_ner\train.conll'
DEV_FILE = r'D:\Project\NLP\data\data_ner\dev.conll'
TEST_FILE = r'D:\Project\NLP\data\data_ner\test.conll'

PREPROCESSOR_PATH = r"D:\Project\NLP\result\crf_preprocessor.pkl"
MODEL_PATH = r'D:\Project\NLP\result\crf_model.pkl'


def main():
    # --- 1. TẢI TẤT CẢ DỮ LIỆU ---
    print("--- 1. Bắt đầu tải dữ liệu ---")
    X_train_sents, y_train_labels = load_conll_data(TRAIN_FILE)
    X_dev_sents, y_dev_labels = load_conll_data(DEV_FILE)
    X_test_sents, y_test_labels = load_conll_data(TEST_FILE)
    
    if not X_train_sents or not X_dev_sents or not X_test_sents:
        print("Lỗi: Không tải được dữ liệu. Vui lòng kiểm tra lại đường dẫn tệp.")
        return
    print("--- Tải dữ liệu hoàn tất ---")

    # --- 2. TIỀN XỬ LÝ ---
    print("\n--- 2. Bắt đầu tiền xử lý ---")
    preprocessor = CRFPreprocessor()

    # Học vocab và trích xuất đặc trưng cho tập TRAIN
    X_train_features = preprocessor.fit_transform(X_train_sents, y_train_labels)
    
    # Chỉ trích xuất đặc trưng cho tập DEV và TEST
    X_dev_features = preprocessor.transform(X_dev_sents)
    X_test_features = preprocessor.transform(X_test_sents)
    
    print("--- Tiền xử lý hoàn tất ---")
    
    # ==========================================================
    # --- (MỚI) IN THÔNG TIN TỪ VỰNG ---
    # ==========================================================
    print("\n--- 2b. Thông tin Từ vựng đã xây dựng ---")
    
    # 1. In thông tin từ vựng (word2idx)
    print(f"Tổng số từ vựng (word2idx): {len(preprocessor.word2idx)}")
    
    # In 20 từ đầu tiên làm mẫu (vì từ vựng có thể rất lớn)
    print("Mẫu 20 từ đầu tiên trong từ vựng:")
    try:
        sample_vocab = list(preprocessor.word2idx.items())[:20]
        for word, idx in sample_vocab:
            print(f"  {word}: {idx}")
    except Exception as e:
        print(f"  Không thể in mẫu: {e}")
        
    # 2. In thông tin nhãn (label2idx)
    print("\nCác nhãn (label2idx):")
    if preprocessor.label2idx:
        for label, idx in preprocessor.label2idx.items():
            print(f"  {label}: {idx}")
    else:
        print("  Chưa có thông tin nhãn.")
    
    print("------------------------------------------")
    # ==========================================================

    # Lưu preprocessor để dùng sau này (ví dụ: trong predict.py)
    preprocessor.save(PREPROCESSOR_PATH)


    # --- 3. HUẤN LUYỆN MÔ HÌNH ---
    print("\n--- 3. Bắt đầu huấn luyện CRF ---")
    crf_model = CRFModel(
        c1=0.1,
        c2=0.1,
        max_iterations=100
    )
    
    crf_model.fit(X_train_features, y_train_labels)
    
    # Lưu model đã huấn luyện
    crf_model.save(MODEL_PATH)

    # --- 4. ĐÁNH GIÁ ---
    print("\n--- 4. Bắt đầu đánh giá ---")
    
    # Lấy danh sách nhãn để báo cáo (loại bỏ 'O')
    report_labels = [lbl for lbl in preprocessor.label2idx.keys() if lbl != 'O']

    print("====================================")
    print("   Kết quả trên tập DEV (Validation)  ")
    print("====================================")
    crf_model.evaluate(X_dev_features, y_dev_labels, report_labels)

    print("\n====================================")
    print("     Kết quả trên tập TEST (Final)    ")
    print("====================================")
    crf_model.evaluate(X_test_features, y_test_labels, report_labels)

if __name__ == "__main__":
    main()