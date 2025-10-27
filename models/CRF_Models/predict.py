# file: predict.py

from preprocessor import CRFPreprocessor
from model import CRFModel

PREPROCESSOR_PATH = r"D:\Project\NLP\result\crf_preprocessor.pkl"
MODEL_PATH = r'D:\Project\NLP\result\crf_model.pkl'

def predict_sentence(text, preprocessor, model):
    """
    Dự đoán nhãn cho một câu (văn bản thô).
    """
    # 1. Tách từ (tokenize) - Giả sử tách từ đơn giản bằng dấu cách
    sentence = text.split()
    print(f"Câu đầu vào: {sentence}")
    
    # 2. Tiền xử lý: Biến câu (list[str]) thành features
    #    Lưu ý: preprocessor.transform nhận vào một list các câu
    X_features = preprocessor.transform([sentence])
    
    # 3. Dự đoán
    #    Lưu ý: model.predict nhận vào list các features
    y_pred = model.predict(X_features)
    
    # 4. In kết quả (y_pred là list[list[str]], ta lấy phần tử đầu tiên)
    return list(zip(sentence, y_pred[0]))

if __name__ == "__main__":
    print("--- Tải các đối tượng đã lưu ---")
    try:
        preprocessor = CRFPreprocessor.load(PREPROCESSOR_PATH)
        model = CRFModel.load(MODEL_PATH)
        print("--- Tải thành công! ---")
        
        # --- THỬ DỰ ĐOÁN ---
        text_1 = "Bệnh nhân bị đau bụng 5 ngày"
        text_2 = "Chỉ định dùng Paracetamol 500 mg"
        
        print("\n--- Dự đoán 1 ---")
        results_1 = predict_sentence(text_1, preprocessor, model)
        print("Kết quả:", results_1)

        print("\n--- Dự đoán 2 ---")
        results_2 = predict_sentence(text_2, preprocessor, model)
        print("Kết quả:", results_2)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp model hoặc preprocessor.")
        print("Vui lòng chạy 'train.py' trước để tạo ra các tệp này.")