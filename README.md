# Animal Text Classifier with DistilBERT

This project uses a fine-tuned DistilBERT model to classify 6 animals

---

## Project Structure

```
Animal-Classifier
├── data/
│   ├── processed/
│   │   └── animal_dataset.csv
│   ├── raw/
│   │   ├── cat_wiki_sentences.csv
│   │   ├── dog_wiki_sentences.csv
│   │   └── …
│   └── dataset_loader.py
│
├── models/
│   ├── distilbert_model.py
│   └── train.py
│
├── saved_models/
│   └── distilbert_animal_classifier.pth
│
├── utils/
│   └── config.py
│
├── app.py             # Flask app for interactive prediction
├── predict.py         # CLI script prediction
├── requirements.txt
└── train.py           # Entry point for training
```

---

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

- Place all raw CSV files inside the data/raw/ directory. Each CSV should contain two columns: description and label.
- Merge all the CSVs by running the code in the Python notebook located at notebooks/text_cleaning.ipynb.
- After running the notebook, a merged file named animal_dataset.csv will be generated in the data/processed/ folder.

### 3. Train the model

```
python train.py
```

- After training is complete, a .pth file named distilbert_animal.pth will be generated.

### 4. Evaluate the model

- Run the script in notebooks/evaluate_model.ipynb

### 5. Make prediction

- Use the CLI script for prediction:

```
python predict.py
```

- The script will prompt you to enter an animal description and output the predicted class.

### 6. Run the flask web application (optional)

- Run the flask app

```
python app.py
```

- By default, it will run at: http://localhost:5050/predict
- User can create a simple website to make a prediction via call and fetch API

---

### Tools & Libraries Used

| Tool/Library      | Purpose                             | License      |
| ----------------- | ----------------------------------- | ------------ |
| **PyTorch**       | Deep learning framework             | BSD-3-Clause |
| **Transformers**  | Pretrained DistilBERT and tokenizer | Apache 2.0   |
| **Datasets**      | Dataset loading and preprocessing   | Apache 2.0   |
| **Scikit-learn**  | Metrics, evaluation, preprocessing  | BSD-3-Clause |
| **Pandas**        | Data manipulation                   | BSD-3-Clause |
| **NumPy**         | Numerical operations                | BSD-3-Clause |
| **TQDM**          | Progress bars                       | MIT          |
| **Matplotlib**    | Visualization                       | PSF License  |
| **Flask**         | REST API backend                    | BSD-3-Clause |
| **Flask-CORS**    | CORS support for Flask              | MIT          |
| **SentencePiece** | Tokenizer model support (if used)   | Apache 2.0   |
