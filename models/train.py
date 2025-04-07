import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.distilbert_model import DistilBERTClassifier
from data.dataset_loader import load_dataset
from utils.config import NUM_CLASSES, LEARNING_RATE, EPOCHS
import os

def train(model, train_loader, val_loader, device):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Train Loss: {total_loss:.4f} | Train Accuracy: {accuracy:.4f}")

        evaluate(model, val_loader, device)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/distilbert_animal_classifier.pth")
    print("\nModel saved to saved_models/distilbert_animal_classifier.pth")

def evaluate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")