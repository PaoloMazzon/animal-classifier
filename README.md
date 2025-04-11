# Animal Classifier (NLP Project)

A lightweight NLP model using DistilBERT to classify animals based on descriptive text. The model was trained on a dataset sourced and processed by our team from Wikipedia and Animalia.bio.

---

## Project Overview

This project demonstrates the use of Natural Language Processing to classify animal species (e.g., dog, cat, bat, snake) using descriptive sentences. It is built using the DistilBERT transformer architecture for text classification.

---

## Model Performance

Evaluation on Validation Set:
- Accuracy: 0.88  
- Precision: 0.87  
- Recall: 0.89  
- F1 Score: 0.88  

_Values are based on validation set using balanced dataset. Reproducibility may vary slightly._

---

## How to Train

```bash
python models/train.py
