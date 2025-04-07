import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from utils.config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

class AnimalDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_dataset(csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)

    unique_labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_to_idx)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['description'], df['label'], test_size=test_size, random_state=42, stratify=df['label']
    )

    train_dataset = AnimalDataset(train_texts.tolist(), train_labels.tolist())
    val_dataset = AnimalDataset(val_texts.tolist(), val_labels.tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, label_to_idx, train_labels.tolist(), val_labels.tolist()