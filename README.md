# NLP SMS Spam Classifier

This project is a complete **end‑to‑end NLP pipeline** for classifying SMS text messages as **spam** or **ham**.  
It was originally developed as a graduate‑level NLP assignment and has been refactored into a clean, local‑runnable repository for professional use.

---

## 1. Project Overview

**Goal:** Build a robust binary classifier for SMS messages using both classic ML and deep learning models, supported by heavy preprocessing, data augmentation, and multiple evaluation tools.

The main script lives in:

```
src/nlp_clean_to_train.py
```

---

## 2. Dataset

This project uses the widely known **SMS Spam Collection Dataset**, often seen in Kaggle and UCI ML resources.

The dataset:

- Column `v1` → label (`ham` or `spam`)
- Column `v2` → message text

The script renames these to:

- `label`
- `sentence`

Place the dataset here:

```
data/spam.csv
```

The dataset is **not included** in the repo.

---

## 3. Pipeline Overview

### 3.1 Preprocessing

The script performs extensive text normalization:

- Contraction expansion
- Acronym expansion
- HTML unescape
- Emoji → text (`emoji.demojize`)
- URL / @username / hashtags handling
- Lowercasing
- Punctuation removal
- Optional number → words (`num2words`)
- Tokenization (NLTK)
- Stopword removal
- POS‑aware lemmatization

### 3.2 Data Balancing

The spam class is augmented using **back‑translation**:

- Spam → French → English  
- Spam → Russian → English  
- Spam → German → English  

This increases variation in spam messages.  
Ham is **downsampled** to match spam after augmentation.

### 3.3 Multiple Text “Views”

Each message is split into:

- Full processed text  
- Noun‑only tokens  
- Verb‑only tokens  

Each view becomes a separate vectorization stream.

### 3.4 Feature Engineering

- TF‑IDF vocabulary
- Keras `Tokenizer` + `pad_sequences` for each view

### 3.5 Models Trained

**Baselines:**

- Multinomial Naive Bayes
- Logistic Regression

**Deep Learning:**

- Several Conv1D architectures with varying kernel sizes
- LSTM
- Bidirectional LSTM (BiLSTM)

### 3.6 Evaluation & Visualization

- Accuracy scores for all models
- Confusion matrices
- Validation accuracy curves
- Validation loss curves
- Word clouds
- Most‑common word bar charts

---

## 4. Project Structure

```
nlp-sms-spam-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── nlp_clean_to_train.py
└── data/
    └── spam.csv (not checked into git)
```

---

## 5. Installation

### Clone repo

```
git clone git@github.com:mannybuff/nlp-sms-spam-classifier.git
cd nlp-sms-spam-classifier
```

### Create environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Add dataset

Place `spam.csv` into `data/`.

---

## 6. Usage

```
python -m src.nlp_clean_to_train
```

or

```
python src/nlp_clean_to_train.py
```

Running the script:

1. Loads dataset  
2. Cleans + augments text  
3. Builds multiple feature views  
4. Trains all baseline + deep models  
5. Displays charts and evaluation metrics  

---

## 7. Future Improvements

- Add HuggingFace transformer baseline (DistilBERT, BERT)  
- Export trained models for inference server  
- Hyperparameter tuning  
- Optional notebook walkthrough  

---

## 8. License

MIT or similar license can be added upon request.
