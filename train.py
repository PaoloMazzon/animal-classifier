from models.train import train
from models.distilbert_model import DistilBERTClassifier
from data.dataset_loader import load_dataset
from utils.config import NUM_CLASSES
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_dataset("data/processed/animal_dataset.csv")

    model = DistilBERTClassifier(num_classes=NUM_CLASSES)
    train(model, train_loader, val_loader, device)