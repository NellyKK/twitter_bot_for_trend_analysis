import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter
import os

# --- NEW IMPORTS FOR DEEP LEARNING ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords

#Global Values and Functions
stop_words = set(stopwords.words('english'))
# Remove negation words - important for sentiment analysis
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard('nor')
stop_words.discard("don")
stop_words.discard("don't")
stop_words.discard("doesn't")
stop_words.discard("didn't")
stop_words.discard("wasn't")
stop_words.discard("weren't")
stop_words.discard("won't")
stop_words.discard("wouldn't")
stop_words.discard("couldn't")
stop_words.discard("shouldn't")
stop_words.discard("isn't")
stop_words.discard("aren't")
stop_words.discard("hasn't")
stop_words.discard("haven't")
stop_words.discard("hadn't")
stop_words.discard("needn't")
stop_words.discard("mustn't")
# Add amp (common in tweets)
stop_words.add('amp')

def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()                          # Standardize case
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"@\w+", "", text)             # Remove mentions
    text = re.sub(r"#", "", text)                # Remove # symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)      # Keep only letters
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

def build_model():
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))) #Add this
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# START OF THE PROGRAM

# ==========================================
# GPU Configuration
# ==========================================
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

# ==========================================
# PROMPT: Load existing model or train new one
# ==========================================
model_path = r"D:\Advanced_Python\project\sentiment_lstm_final_ntlk.keras"

if os.path.exists(model_path):
    choice = input("Saved model found. Load it and skip training? (y/n): ")
    if choice.lower() == 'y':
        final_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully! Skipping Steps 1-6.")
        # You can add prediction code here
        exit()  # or replace with prediction logic

# ==========================================
# STEP 1: Load the dataset
# ==========================================
file_path = r"D:\Advanced_Python\project\TweetsWithLabels.csv"

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found.")

# ==========================================
# STEP 2: Data Verification and Cleaning
# ==========================================
# Limit rows for processing speed
# df = df.head(60000)
print(df.head())

#Verify null/NaN data
is_null = df.isnull().sum()
print(is_null)
# textID           0
# text             1
# selected_text    1
# sentiment        0

#Print null/NaN data
print(df[df['text'].isna() | df['selected_text'].isna()])
#          textID text selected_text sentiment
# 314  fdb77c3752  NaN           NaN   neutral

#Drop null/NaN data
df = df.dropna(subset=['text'])

#Clean text
df["clean_text"] = df["text"].apply(clean_tweet)
print(df.head())
#        textID  ...                                         clean_text
# 0  cb774db0d1  ...                            id responded were going
# 1  549e992a42  ...                  sooo sad will miss here san diego
# 2  088c60f138  ...                                   boss is bullying
# 3  9642c003ef  ...                              interview leave alone
# 4  358bd9e861  ...  sons why couldnt they put them releases alread...

# Verify Sentiment Distribution
print(df['sentiment'].value_counts())

# sns.countplot(x='sentiment', data=df)
# plt.title("Sentiment Distribution")
# plt.show()

# sentiment
# neutral     11117
# positive     8582
# negative     7781
# -> Imbalanced

# ==========================================
# STEP 3: PREPARE DATA FOR LSTM - Transforms raw text and labels into numerical arrays for LSTM
# ==========================================
print("\n>>> Preparing data for Deep Learning (LSTM)...")

MAX_NB_WORDS = 5000
# MAX_NB_WORDS = 10000
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
# STEP 4: Train and Test LSTM w/ 10-FOLD CROSS VALIDATION, Stratification
# ==========================================
print(">>> Starting 10-Fold Cross Validation...")

NUM_FOLDS = 10
kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_accuracies = []
fold_losses = []

# Function to build a fresh model for each fold
# def build_model():
#     model = Sequential()
#     model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
#     model.add(SpatialDropout1D(0.2))
#     # model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#     model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# Run each fold
all_y_true = []
all_y_pred = []

for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y_integer), 1):
    print(f"\n--- Fold {fold_num}/{NUM_FOLDS} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]

    # Build a fresh model for each fold
    model = build_model()

    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=15,
              batch_size=128,
              validation_data=(X_test, y_test),
              callbacks=[early_stop],
              verbose=1)
    
    # model.fit(X_train, y_train,
    #           epochs=5,
    #           batch_size=128,
    #           validation_data=(X_test, y_test),
    #           verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    all_y_true.extend(y_true_classes)
    all_y_pred.extend(y_pred_classes)
    
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
print(f"  Mean Error Rate: {1 - np.mean(fold_accuracies):.4f}")

# ZeroR Baseline
zeror_accuracy = df['sentiment'].value_counts().max() / len(df)
print(f"  ZeroR Baseline:  {zeror_accuracy:.4f}")
print(f"  Improvement over ZeroR: {np.mean(fold_accuracies) - zeror_accuracy:.4f}")

print("\nOverall Classification Report (All Folds Combined):")
print(classification_report(all_y_true, all_y_pred, 
      target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(all_y_true, all_y_pred))
print("=" * 50)

# ==========================================
# STEP 6: VISUALIZE FOLD RESULTS
# ==========================================
# plt.figure(figsize=(12, 5))

# # Accuracy per fold
# plt.subplot(1, 2, 1)
# plt.bar(range(1, NUM_FOLDS + 1), fold_accuracies, color='steelblue')
# plt.axhline(y=np.mean(fold_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(fold_accuracies):.4f}')
# plt.xlabel('Fold')
# plt.ylabel('Accuracy')
# plt.title('Accuracy per Fold')
# plt.xticks(range(1, NUM_FOLDS + 1))
# plt.legend()

# # Loss per fold
# plt.subplot(1, 2, 2)
# plt.bar(range(1, NUM_FOLDS + 1), fold_losses, color='coral')
# plt.axhline(y=np.mean(fold_losses), color='red', linestyle='--', label=f'Mean: {np.mean(fold_losses):.4f}')
# plt.xlabel('Fold')
# plt.ylabel('Loss')
# plt.title('Loss per Fold')
# plt.xticks(range(1, NUM_FOLDS + 1))
# plt.legend()

# plt.tight_layout()
# plt.show()

# ==========================================
# STEP 7: SAVE / LOAD MODEL
# ==========================================

# Train final model on ALL data (not just one fold)
# print("\n>>> Training final model on all data...")
# final_model = build_model()

# # final_model.fit(X, y_onehot, epochs=5, batch_size=128, verbose=1)

# early_stop_final = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# final_model.fit(X, y_onehot,
#                 epochs=15,
#                 batch_size=128,
#                 callbacks=[early_stop_final],
#                 verbose=1)

print("\n>>> Training final model on all data...")
final_model = build_model()
final_model.fit(X, y_onehot,
                epochs=3,
                batch_size=128,
                verbose=1)

# Save model
save_path = r"D:\Advanced_Python\project\sentiment_lstm_final_ntlk.keras"
final_model.save(save_path)
print(f"Model saved to {save_path}")

# Save fold results for later visualization
results = pd.DataFrame({
    'Fold': range(1, NUM_FOLDS + 1),
    'Accuracy': fold_accuracies,
    'Loss': fold_losses
})
results_path = r"D:\Advanced_Python\project\fold_results_ntlk.csv"
results.to_csv(results_path, index=False)
print(f"Fold results saved to {results_path}")