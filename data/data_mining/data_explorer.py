import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path
import logging

class ViMedNERExplorer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.sentences = []
        self.labels = []
        
    def load_conll_format(self, file_path):
        """Đọc dữ liệu định dạng CoNLL-U"""
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':  # Kết thúc câu
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0]
                        label = parts[1]
                        current_sentence.append(word)
                        current_labels.append(label)
        
        # Thêm câu cuối nếu có
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
            
        return sentences, labels
    
    def explore_dataset(self):
        """Khám phá thống kê cơ bản của dataset"""
        train_sentences, train_labels = self.load_conll_format(f"{self.data_path}/train.conll")
        val_sentences, val_labels = self.load_conll_format(f"{self.data_path}/dev.conll")
        test_sentences, test_labels = self.load_conll_format(f"{self.data_path}/test.conll")
        
        print("=== THỐNG KÊ CƠ BẢN ===")
        print(f"Số câu train: {len(train_sentences)}")
        print(f"Số câu val: {len(val_sentences)}")
        print(f"Số câu test: {len(test_sentences)}")
        
        # Thống kê độ dài câu
        train_lengths = [len(sent) for sent in train_sentences]
        print(f"Độ dài câu trung bình (train): {np.mean(train_lengths):.2f}")
        print(f"Độ dài câu min/max (train): {min(train_lengths)}/{max(train_lengths)}")
        
        # Thống kê nhãn
        all_labels = [label for labels in train_labels for label in labels]
        label_counts = Counter(all_labels)
        
        print("\n=== PHÂN PHỐI NHÃN ===")
        for label, count in label_counts.most_common():
            print(f"{label}: {count} ({count/len(all_labels)*100:.2f}%)")

        return {
            'train': (train_sentences, train_labels),
            'val': (val_sentences, val_labels),
            'test': (test_sentences, test_labels),
            'label_counts': label_counts
        }
        
    def visualize_statistics(self, data_dict):
        """Tạo các biểu đồ thống kê riêng biệt"""

        # 1. Biểu đồ phân phối độ dài câu
        train_lengths = [len(sent) for sent in data_dict['train'][0]]
        plt.figure(figsize=(8,6))
        plt.hist(train_lengths, bins=50, alpha=0.7)
        plt.title('Phân phối độ dài câu')
        plt.xlabel('Số từ')
        plt.ylabel('Tần suất')
        plt.savefig('sentence_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Biểu đồ phân phối nhãn
        labels = list(data_dict['label_counts'].keys())
        counts = list(data_dict['label_counts'].values())
        plt.figure(figsize=(8,6))
        plt.bar(labels, counts)
        plt.title('Phân phối nhãn')
        plt.xlabel('Nhãn')
        plt.ylabel('Số lượng')
        plt.xticks(rotation=45)
        plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Biểu đồ phân phối loại thực thể
        entity_types = {}
        for label in labels:
            if label != 'O':
                entity_type = label.split('-')[-1] if '-' in label else label
                entity_types[entity_type] = entity_types.get(entity_type, 0) + data_dict['label_counts'][label]
        plt.figure(figsize=(8,6))
        plt.pie(entity_types.values(), labels=entity_types.keys(), autopct='%1.1f%%')
        plt.title('Phân phối loại thực thể')
        plt.savefig('entity_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Biểu đồ top từ xuất hiện nhiều nhất
        all_words = [word for sent in data_dict['train'][0] for word in sent]
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(20)
        words, word_freq = zip(*top_words)



        plt.figure(figsize=(8,6))
        plt.barh(words, word_freq)
        plt.title('Top 20 từ xuất hiện nhiều nhất')
        plt.xlabel('Tần suất')
        plt.gca().invert_yaxis()  # để từ xuất hiện nhiều nhất nằm trên cùng
        plt.savefig('top20_words.png', dpi=300, bbox_inches='tight')
        plt.show()

# # Sử dụng
explorer = ViMedNERExplorer(r"D:\Project\NLP\data\data_ner")
data_dict = explorer.explore_dataset()
explorer.visualize_statistics(data_dict)