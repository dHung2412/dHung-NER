from torch.utils.data import Dataset, DataLoader
import torch

class ViMedNERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=256, label2idx=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2idx = label2idx if label2idx else self._build_label_vocab(labels)
        
    def _build_label_vocab(self, all_labels):
        """Xây dựng từ điển nhãn"""
        unique_labels = set()
        for label_seq in all_labels:
            unique_labels.update(label_seq)
        unique_labels = sorted(list(unique_labels))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx] if self.labels else None
        
        # Tokenize và align labels với subwords
        tokenized = self.tokenize_and_align_labels(sentence, labels)
        
        return {
            'input_ids': torch.tensor(tokenized['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(tokenized['labels'], dtype=torch.long) if labels else None
        }
    
    def tokenize_and_align_labels(self, sentence, labels=None):
        """Tokenize và align labels với subwords"""
        # Join words thành câu
        text = ' '.join(sentence)
        
        # Tokenize
        tokenized_inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None,
            is_split_into_words=False
        )
        
        # Align labels nếu có
        if labels is not None:
            # Tokenize từng từ để biết boundary
            word_ids = []
            current_word_id = 0
            tokens = self.tokenizer.tokenize(text)
            
            # Tạo word_ids mapping
            for i, token_id in enumerate(tokenized_inputs['input_ids']):
                if i == 0:  # [CLS]
                    word_ids.append(None)
                elif i == len(tokenized_inputs['input_ids']) - 1:  # [SEP]
                    word_ids.append(None)
                elif tokenized_inputs['input_ids'][i] == self.tokenizer.pad_token_id:  # [PAD]
                    word_ids.append(None)
                else:
                    # Tìm word tương ứng
                    token = self.tokenizer.decode([token_id])
                    if current_word_id < len(sentence):
                        word_ids.append(current_word_id)
                        # Kiểm tra xem có phải là token cuối của word không
                        if not token.startswith('##') and current_word_id < len(sentence) - 1:
                            # Kiểm tra xem token tiếp theo có phải là subword không
                            next_token_idx = i + 1
                            if next_token_idx < len(tokenized_inputs['input_ids']):
                                next_token = self.tokenizer.decode([tokenized_inputs['input_ids'][next_token_idx]])
                                if not next_token.startswith('##'):
                                    current_word_id += 1
                    else:
                        word_ids.append(None)
            
            # Align labels
            aligned_labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # Ignore trong loss
                elif word_idx != previous_word_idx:
                    # Lấy label cho từ này
                    if word_idx < len(labels):
                        aligned_labels.append(self.label2idx[labels[word_idx]])
                    else:
                        aligned_labels.append(-100)
                else:
                    # Subword token, sử dụng -100 để ignore
                    aligned_labels.append(-100)
                previous_word_idx = word_idx
            
            tokenized_inputs['labels'] = aligned_labels
        
        return tokenized_inputs