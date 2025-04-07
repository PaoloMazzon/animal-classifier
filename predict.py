import torch
from transformers import DistilBertTokenizer
from models.distilbert_model import DistilBERTClassifier
from utils.config import MODEL_NAME, NUM_CLASSES, MAX_LENGTH

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBERTClassifier(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("saved_models/distilbert_animal_classifier.pth", map_location='cpu'))
model.eval()

id_to_label = {
    0: "bat",
    1: "cat",
    2: "dog",
    3: "snake"
}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

sample = "This animal flies at night and uses echolocation."
print(predict(sample))
predicted_id = predict(sample)
print("Predicted class:", id_to_label[predicted_id])