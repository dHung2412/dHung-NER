import json
import re

# ====== Đường dẫn ======
input_path = r"D:\Project\NLP\data\data_ner\ViMedNER.txt"
output_path = r"D:\Project\NLP\data\data_ner\ViMedNER_converted.conll"

# ====== Bảng ánh xạ nhãn ======
entity_mapping = {
    "DIAGNOSTIC": "bien_phap_chan_doan",
    "TREATMENT": "bien_phap_dieu_tri",
    "CAUSE": "nguyen_nhan_benh",
    "DISEASE": "ten_benh",
    "SYMPTOM": "trieu_chung_benh"
}


def tokenize(text):
    """
    Tokenize cơ bản theo khoảng trắng và dấu câu, 
    có thể thay bằng underthesea nếu cần chính xác hơn.
    """
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    offsets = []
    start = 0
    for token in tokens:
        start = text.find(token, start)
        offsets.append((start, start + len(token)))
        start += len(token)
    return tokens, offsets


def json_to_bio(text, labels):
    """
    Chuyển 1 câu + nhãn sang BIO-format theo token.
    """
    tokens, offsets = tokenize(text)
    tags = ["O"] * len(tokens)

    for start, end, entity in labels:
        entity_vn = entity_mapping.get(entity, entity)
        for i, (tok_start, tok_end) in enumerate(offsets):
            # Kiểm tra token có nằm trong vùng nhãn
            if tok_end <= start or tok_start >= end:
                continue
            if tok_start >= start and tok_end <= end:
                # Token đầu tiên trong entity
                if tags[i] == "O":
                    if (i == 0) or tags[i-1] == "O":
                        tags[i] = f"B-{entity_vn}"
                    else:
                        tags[i] = f"I-{entity_vn}"
                else:
                    tags[i] = f"I-{entity_vn}"

    # Gộp token + tag
    bio_lines = [f"{tok}\t{tag}" for tok, tag in zip(tokens, tags)]
    return bio_lines


with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        text = sample["text"]
        labels = sample.get("label", [])

        bio_lines = json_to_bio(text, labels)
        outfile.write("\n".join(bio_lines))
        outfile.write("\n\n")

print(f"✅ Đã chuyển xong! File lưu tại: {output_path}")
