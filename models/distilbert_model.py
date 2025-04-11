import torch
import torch.nn as nn
from transformers import DistilBertModel

class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.dropout = nn.Dropout(0.3)  # Dropout layer removed
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # pooled_output = self.dropout(pooled_output)  # Dropout application removed
        logits = self.classifier(pooled_output)
        return logits