from models.train import train
from models.distilbert_model import DistilBERTClassifier
from data.dataset_loader import load_dataset
from utils.config import NUM_CLASSES
import torch
import pandas as pd

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, label_to_idx, train_labels, val_labels = load_dataset("data/processed/animal_dataset.csv")

    print("Class distribution in training set:")
    print(pd.Series(train_labels).value_counts())

    print("Class distribution in validation set:")
    print(pd.Series(val_labels).value_counts())

    model = DistilBERTClassifier(num_classes=NUM_CLASSES)

# 🔒 Freeze all encoder layers
    for param in model.encoder.parameters():
        param.requires_grad = False

    # 🔓 Unfreeze only the last 2 transformer layers (layer.4 and layer.5)
    for name, param in model.encoder.named_parameters():
        if "transformer.layer.4" in name or "transformer.layer.5" in name:
            param.requires_grad = True

    train(model, train_loader, val_loader, device)