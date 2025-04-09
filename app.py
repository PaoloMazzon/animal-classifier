from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer
from models.distilbert_model import DistilBERTClassifier
from utils.config import MODEL_NAME, NUM_CLASSES, MAX_LENGTH

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBERTClassifier(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("saved_models/distilbert_animal_classifier.pth", map_location='cpu'))
model.eval()

# Class label mapping
id_to_label = {
    0: "bat",
    1: "cat",
    2: "dog",
    3: "snake"
}

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No input text provided'}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        predicted_class = torch.argmax(logits, dim=1).item()

    return jsonify({
        'prediction_id': predicted_class,
        'prediction_label': id_to_label[predicted_class]
    })

if __name__ == '__main__':
    app.run(host="localhost", port=5050, debug=True)