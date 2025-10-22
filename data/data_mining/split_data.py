import os
import random

# ====== CẤU HÌNH ======
INPUT_FILE = r"D:\Project\NLP\data\data_ner\ViMedNER_converted.conll"
OUTPUT_DIR = os.path.dirname(INPUT_FILE)
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 42
# ======================

def read_conll_file(file_path):
    """Đọc dữ liệu conll thành danh sách các câu (list of sentences)"""
    sentences = []
    current_sentence = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        if current_sentence:  # câu cuối cùng
            sentences.append(current_sentence)
    return sentences

def write_conll_file(sentences, file_path):
    """Ghi danh sách các câu ra file conll"""
    with open(file_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for line in sent:
                f.write(line + "\n")
            f.write("\n")

def split_data(sentences, train_ratio, dev_ratio, test_ratio):
    """Chia dữ liệu theo tỷ lệ"""
    total = len(sentences)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)

    random.seed(SEED)
    random.shuffle(sentences)

    train_data = sentences[:train_size]
    dev_data = sentences[train_size:train_size + dev_size]
    test_data = sentences[train_size + dev_size:]
    return train_data, dev_data, test_data

def main():
    sentences = read_conll_file(INPUT_FILE)
    print(f"Đọc {len(sentences)} câu từ file gốc.")

    train_data, dev_data, test_data = split_data(sentences, TRAIN_RATIO, DEV_RATIO, TEST_RATIO)

    train_path = os.path.join(OUTPUT_DIR, "train.conll")
    dev_path = os.path.join(OUTPUT_DIR, "dev.conll")
    test_path = os.path.join(OUTPUT_DIR, "test.conll")

    write_conll_file(train_data, train_path)
    write_conll_file(dev_data, dev_path)
    write_conll_file(test_data, test_path)

    print(f"✅ Đã chia dữ liệu:")
    print(f"  - Train: {len(train_data)} câu → {train_path}")
    print(f"  - Dev:   {len(dev_data)} câu → {dev_path}")
    print(f"  - Test:  {len(test_data)} câu → {test_path}")

if __name__ == "__main__":
    main()
