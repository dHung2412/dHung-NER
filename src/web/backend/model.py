import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PhoBERTForNER(nn.Module):
    def __init__(self, model_name: str = "vinai/phobert-base", num_labels: int = 11, dropout: float = 0.1):
        super(PhoBERTForNER, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load PhoBERT model
        self.phobert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear classifier layer
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classifier layer"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Hàm forward đã được rút gọn, chỉ trả về logits
        """
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the last hidden state
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Get logits from classifier
        logits = self.classifier(sequence_output)
        
        # Chỉ trả về logits (không cần loss)
        return {
            'logits': logits
        }
    
    def predict(self, input_ids, attention_mask=None):
        """Hàm dự đoán (predict)"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions