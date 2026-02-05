import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter

# --- NEW IMPORTS FOR DEEP LEARNING ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
file_path = r"C:\Users\User\Downloads\tweets_v8.csv"
print(">>> Loading dataset...")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found. Creating dummy data.")
    df = pd.DataFrame({'text': ["I love this!", "This is bad", "Neutral statement"] * 100})

# Limit rows for processing speed
df = df.head(60000)

# 2. Define Stop Words
stop_words = {"the", "i", "to", "a", "and", "of", "in", "is", "it", "for", "you", "on", "this", "that", "my", "with", "are", "be", "at", "me", "have", "so", "but", "was", "not", "just", "im", "your", "like", "all", "do", "we", "can", "from", "about", "an", "or", "has", "what", "if", "up", "out", "amp", "don", "t"}

# 3. Data Cleaning
def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace("squid game", "squidgame").replace("squidgamenetflix", "squidgame")
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

print(">>> Cleaning tweets and calculating sentiment...")
df["clean_text"] = df["text"].apply(clean_tweet)

# Generate labels using TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0: return "Positive"
    elif polarity < 0: return "Negative"
    else: return "Neutral"

df["sentiment"] = df["clean_text"].apply(get_sentiment)

# ==========================================
# STEP 3: PREPARE DATA FOR LSTM
# ==========================================
print("\n>>> Preparing data for Deep Learning (LSTM)...")

MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(df['clean_text'].values)
X = tokenizer.texts_to_sequences(df['clean_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# Encode Labels
label_encoder = LabelEncoder()
y_integer = label_encoder.fit_transform(df['sentiment'])
y_onehot = to_categorical(y_integer)

# ==========================================
# STEP 4: 10-FOLD CROSS VALIDATION
# ==========================================
print(">>> Starting 10-Fold Cross Validation...")

NUM_FOLDS = 10
kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_accuracies = []
fold_losses = []

# Function to build a fresh model for each fold
def build_model():
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Run each fold
for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y_integer), 1):
    print(f"\n--- Fold {fold_num}/{NUM_FOLDS} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]

    # Build a fresh model for each fold
    model = build_model()

    # Train
    model.fit(X_train, y_train,
              epochs=5,
              batch_size=128,
              validation_data=(X_test, y_test),
              verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {fold_num} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    fold_accuracies.append(accuracy)
    fold_losses.append(loss)

# ==========================================
# STEP 5: RESULTS SUMMARY
# ==========================================
print("\n" + "=" * 50)
print("10-FOLD CROSS VALIDATION RESULTS")
print("=" * 50)
for i in range(NUM_FOLDS):
    print(f"  Fold {i+1:2d}: Accuracy = {fold_accuracies[i]:.4f}, Loss = {fold_losses[i]:.4f}")
print("-" * 50)
print(f"  Mean Accuracy:  {np.mean(fold_accuracies):.4f}")
print(f"  Std Accuracy:   {np.std(fold_accuracies):.4f}")
print(f"  Mean Loss:      {np.mean(fold_losses):.4f}")
print("=" * 50)

# ==========================================
# STEP 6: VISUALIZE FOLD RESULTS
# ==========================================
plt.figure(figsize=(12, 5))

# Accuracy per fold
plt.subplot(1, 2, 1)
plt.bar(range(1, NUM_FOLDS + 1), fold_accuracies, color='steelblue')
plt.axhline(y=np.mean(fold_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(fold_accuracies):.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy per Fold')
plt.xticks(range(1, NUM_FOLDS + 1))
plt.legend()

# Loss per fold
plt.subplot(1, 2, 2)
plt.bar(range(1, NUM_FOLDS + 1), fold_losses, color='coral')
plt.axhline(y=np.mean(fold_losses), color='red', linestyle='--', label=f'Mean: {np.mean(fold_losses):.4f}')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Loss per Fold')
plt.xticks(range(1, NUM_FOLDS + 1))
plt.legend()

plt.tight_layout()
plt.show()
