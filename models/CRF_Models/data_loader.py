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
                
                if not line:
                    if current_sentence: 
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split() 
                    if len(parts) >= 2:
                        current_sentence.append(parts[0])
                        current_labels.append(parts[1])
                    else:
                        print(f"Bỏ qua dòng định dạng không hợp lệ: {line}")
        
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