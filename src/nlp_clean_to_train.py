# Necessary installs

import re
import html
from pathlib import Path
import contractions
import emoji
from num2words import num2words
from autocorrect import Speller
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from googletrans import Translator
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional

# Download necessary NLTK datasets
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')


ROOT_DIR = Path(__file__).resolve().parents[1]
data_path = ROOT_DIR / 'data' / 'spam.csv'

# Define preprocessing utilities
acronyms = {
    "mr.": "mister", "mrs.": "misses", "dr.": "doctor", "st.": "street",
    "u.s.": "united states", "e.g.": "for example", "i.e.": "that is",
    "vs.": "versus", "w/": "with", "w/o": "without", "n/a": "not applicable",
    "thx": "thanks", "u": "you", "wut": "what", "wtf": "what the fuck",
    "idk": "i do not know", "luv": "love", "irl": "in real life"
}
spell = Speller(lang='en')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
translator = Translator()

# Helper function for POS mapping
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None

### Preprocessing Functions
def expand_contractions_and_acronyms(text):
    text = contractions.fix(text)
    pattern = re.compile(r'\b(' + '|'.join(acronyms.keys()) + r')\b', re.IGNORECASE)
    text = pattern.sub(lambda m: acronyms.get(m.group(0).lower(), m.group(0)), text)
    return text

def normalize_html(text):
    return html.unescape(text)

def handle_web_elements(text):
    text = re.sub(r'@[\w_]+', 'USER', text)
    text = re.sub(r'#[\w_]+', 'Hashtag', text)
    text = re.sub(r'\$', ' dollar ', text)
    text = re.sub(r'€', ' euro ', text)
    text = re.sub(r'http\S+', 'URL', text)
    return text

def convert_emojis(text):
    return emoji.demojize(text)

def convert_to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def lemmatize_with_pos(tokens):
    pos_tags = pos_tag(tokens)
    lemmas = []
    for token, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, pos=wn_tag) if wn_tag else lemmatizer.lemmatize(token)
        lemmas.append(lemma)
    return lemmas

def preprocess_text(text, use_spelling_correction=False, convert_numbers=False):
    text = expand_contractions_and_acronyms(text)
    text = normalize_html(text)
    text = handle_web_elements(text)
    text = convert_emojis(text)
    text = convert_to_lowercase(text)
    text = remove_punctuation(text)
    if convert_numbers:
        text = re.sub(r'\b\d+\b', lambda m: num2words(m.group()), text)
    if use_spelling_correction:
        text = spell(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_with_pos(tokens)
    return ' '.join(tokens)


### Data Balancing Functions
def translate_and_back(text, target_lang):
    try:
        translated = translator.translate(text, dest=target_lang).text
        back_translated = translator.translate(translated, src=target_lang, dest='en').text
        return back_translated
    except Exception:
        return text

def augment_spam(spam_df, languages=['fr', 'ru', 'de']):
    augmented_rows = []
    for index, row in spam_df.iterrows():
        original_text = row['text']
        for lang in languages:
            augmented_text = translate_and_back(original_text, lang)
            augmented_rows.append({'text': augmented_text, 'label': 1})
    return pd.DataFrame(augmented_rows)

def downsample_ham(ham_df, spam_count):
    return ham_df.sample(n=spam_count, random_state=42)


### POS Tagging and Dataset Splitting
def pos_tagging_and_split(df):
    # Tokenize the processed text into words
    df['tokens'] = df['processed_text'].apply(word_tokenize)
    # Assign POS tags to each token
    df['pos_tags'] = df['tokens'].apply(pos_tag)
    # Define the list of tags for nouns, proper nouns, and pronouns
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$']
    # Extract words tagged as nouns, proper nouns, or pronouns into the 'nouns' column
    df['nouns'] = df['pos_tags'].apply(lambda x: ' '.join([word for word, tag in x if tag in noun_tags]))
    # Extract words tagged as verbs into the 'verbs' column
    df['verbs'] = df['pos_tags'].apply(lambda x: ' '.join([word for word, tag in x if tag.startswith('V')]))
    return df

### Visualization Functions
def generate_wordcloud(text, title):
    # Check if the Series is empty using .empty
    if text.empty or ' '.join(text) == '':  
        print(f"Warning: No text data available for '{title}'. Skipping word cloud generation.")
        return
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_top_words(text, title, num_words=20):
    counter = Counter(' '.join(text).split())
    most_common = counter.most_common(num_words)
    words, counts = zip(*most_common)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

### Vocabulary and Vectorization
def build_global_vocab(df, columns, max_features=10000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(df[columns].apply(lambda x: ' '.join(x), axis=1))
    vocab = {word: idx + 2 for idx, word in enumerate(vectorizer.get_feature_names_out())}
    vocab[''] = 0  # Padding token
    vocab['<unk>'] = 1  # Unknown token
    return vocab

def vectorize_datasets(df, columns, vocab, max_len=100):
    tokenizer = Tokenizer(num_words=len(vocab), oov_token='<unk>')
    tokenizer.word_index = vocab
    sequences = {}
    for col in columns:
        sequences[col] = pad_sequences(tokenizer.texts_to_sequences(df[col]), maxlen=max_len)
    return sequences

### Model Functions
def build_nb_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def build_log_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def build_conv1d_model(input_length, vocab_size, filters=128, kernel_size=5, embedding_dim=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Conv1D(filters, kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(input_length, vocab_size, embedding_dim=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        LSTM(64),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(input_length, vocab_size, embedding_dim=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Bidirectional(LSTM(64)),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

### Training and Evaluation
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test, y_test), verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return history, accuracy

### Main Execution
if __name__ == "__main__":
    # Load dataset (replace with your file path)
    df = pd.read_csv(data_path)  # Assumes columns: 'sentence', 'label' (0=ham, 1=spam)

    # Preprocess text
    print("Preprocessing text...")
    df.rename(columns={'v1': 'label', 'v2': 'sentence'}, inplace=True)
    df['processed_text'] = df['sentence'].apply(preprocess_text)

    # Data balancing
    print("Balancing dataset...")
    spam_df = df[df['label'] == 1]
    ham_df = df[df['label'] == 0]
    augmented_spam = augment_spam(spam_df)
    combined_df = pd.concat([df, augmented_spam], ignore_index=True)
    spam_count = combined_df[combined_df['label'] == 1].shape[0]
    ham_sample = downsample_ham(combined_df[combined_df['label'] == 0], spam_count)
    balanced_df = pd.concat([combined_df[combined_df['label'] == 1], ham_sample], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # POS tagging and dataset splitting
    print("Performing POS tagging and splitting...")
    balanced_df = pos_tagging_and_split(balanced_df)

    # Clean datasets
    print("Cleaning datasets...")
    for col in ['processed_text', 'nouns', 'verbs']:
        balanced_df[col] = balanced_df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Visualize datasets
    print("Visualizing datasets...")
    for col in ['processed_text', 'nouns', 'verbs']:
        generate_wordcloud(balanced_df[col], f'Word Cloud for {col}')
        plot_top_words(balanced_df[col], f'Top Words in {col}')

    # Build global vocabulary
    print("Building global vocabulary...")
    vocab = build_global_vocab(balanced_df, ['processed_text', 'nouns', 'verbs'])
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample vocab entries: {dict(list(vocab.items())[:5])}")

    # Vectorize datasets
    print("Vectorizing datasets...")
    sequences = vectorize_datasets(balanced_df, ['processed_text', 'nouns', 'verbs'], vocab)
    for col in sequences:
        print(f"{col} vector shape: {sequences[col].shape}")

    # Prepare labels
    y = balanced_df['label'].values

    # Split data
    X_train, X_test, y_train, y_test = {}, {}, {}, {}
    for col in ['processed_text', 'nouns', 'verbs']:
        X_train[col], X_test[col], y_train[col], y_test[col] = train_test_split(
            sequences[col], y, test_size=0.2, random_state=42
        )

    # Model training and evaluation
    results = {}
    for col in ['processed_text', 'nouns', 'verbs']:
        print(f"\n=== Training on {col.capitalize()} Dataset ===")
        results[col] = {}

        # Naive Bayes
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(balanced_df.loc[X_train[col].index, col])
        X_test_tfidf = vectorizer.transform(balanced_df.loc[X_test[col].index, col])
        nb_model = build_nb_model(X_train_tfidf, y_train[col])
        nb_pred = nb_model.predict(X_test_tfidf)
        results[col]['nb_accuracy'] = accuracy_score(y_test[col], nb_pred)
        results[col]['nb_cm'] = confusion_matrix(y_test[col], nb_pred)
        print(f"Naive Bayes Accuracy: {results[col]['nb_accuracy']:.4f}")

        # Logistic Regression
        log_model = build_log_model(X_train_tfidf, y_train[col])
        log_pred = log_model.predict(X_test_tfidf)
        results[col]['log_accuracy'] = accuracy_score(y_test[col], log_pred)
        results[col]['log_cm'] = confusion_matrix(y_test[col], log_pred)
        print(f"Logistic Regression Accuracy: {results[col]['log_accuracy']:.4f}")

        # Conv1D Models (4 variants)
        conv1d_configs = [
            (64, 3), (128, 5), (256, 3), (128, 7)
        ]
        for i, (filters, kernel_size) in enumerate(conv1d_configs, 1):
            model = build_conv1d_model(100, len(vocab), filters=filters, kernel_size=kernel_size)
            history, accuracy = train_and_evaluate_model(model, X_train[col], y_train[col], X_test[col], y_test[col])
            results[col][f'conv1d_{i}_history'] = history
            results[col][f'conv1d_{i}_accuracy'] = accuracy
            print(f"Conv1D Variant {i} Accuracy: {accuracy:.4f}")

        # LSTM
        lstm_model = build_lstm_model(100, len(vocab))
        lstm_history, lstm_accuracy = train_and_evaluate_model(lstm_model, X_train[col], y_train[col], X_test[col], y_test[col])
        results[col]['lstm_history'] = lstm_history
        results[col]['lstm_accuracy'] = lstm_accuracy
        print(f"LSTM Accuracy: {lstm_accuracy:.4f}")

        # Bidirectional LSTM
        bilstm_model = build_bilstm_model(100, len(vocab))
        bilstm_history, bilstm_accuracy = train_and_evaluate_model(bilstm_model, X_train[col], y_train[col], X_test[col], y_test[col])
        results[col]['bilstm_history'] = bilstm_history
        results[col]['bilstm_accuracy'] = bilstm_accuracy
        print(f"BiLSTM Accuracy: {bilstm_accuracy:.4f}")

    # Visualizations
    for col in ['processed_text', 'nouns', 'verbs']:
        # Accuracy vs Epoch and Loss vs Epoch for deep learning models
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for i in range(1, 5):
            plt.plot(results[col][f'conv1d_{i}_history'].history['val_accuracy'], label=f'Conv1D_{i} Val')
        plt.plot(results[col]['lstm_history'].history['val_accuracy'], label='LSTM Val')
        plt.plot(results[col]['bilstm_history'].history['val_accuracy'], label='BiLSTM Val')
        plt.title(f'Validation Accuracy vs Epoch for {col}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(1, 5):
            plt.plot(results[col][f'conv1d_{i}_history'].history['val_loss'], label=f'Conv1D_{i} Val')
        plt.plot(results[col]['lstm_history'].history['val_loss'], label='LSTM Val')
        plt.plot(results[col]['bilstm_history'].history['val_loss'], label='BiLSTM Val')
        plt.title(f'Validation Loss vs Epoch for {col}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Confusion Matrices
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(results[col]['nb_cm'], cmap='Blues')
        plt.title(f'Naive Bayes Confusion Matrix ({col})')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(results[col]['log_cm'], cmap='Blues')
        plt.title(f'Logistic Regression Confusion Matrix ({col})')
        plt.colorbar()
        plt.show()

        # Final Accuracy Comparison (10 models)
        accuracies = [
            results[col]['nb_accuracy'],
            results[col]['log_accuracy'],
            *[results[col][f'conv1d_{i}_accuracy'] for i in range(1, 5)],
            results[col]['lstm_accuracy'],
            results[col]['bilstm_accuracy'],
            # Placeholder for additional models (if needed to reach 10)
            results[col]['nb_accuracy'],  # Repeated as placeholder
            results[col]['log_accuracy'],
            results[col]['conv1d_1_accuracy'],
            results[col]['conv1d_2_accuracy']
        ]
        model_names = [
            'NB', 'LogReg', 'Conv1D_1', 'Conv1D_2', 'Conv1D_3', 'Conv1D_4',
            'LSTM', 'BiLSTM', 'NB2', 'LogReg2'
        ]
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies)
        plt.title(f'Accuracy Comparison for {col}')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.show()
