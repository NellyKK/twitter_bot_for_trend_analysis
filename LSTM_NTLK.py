import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import re
# from textblob import TextBlob
# from collections import Counter
import os

# --- NEW IMPORTS FOR DEEP LEARNING ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# Global Values
# ==========================================
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# *** Handling Negations - Kept negation words in stop words
stop_words = set(stopwords.words('english'))
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

# ==========================================
# Functions
# ==========================================

# *** Text Cleaning and Lemmatization
def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()                          # Standardize case
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"@\w+", "", text)             # Remove mentions
    text = re.sub(r"#", "", text)                # Remove # symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)      # Keep only letters
    words = text.split()
    words = [lemmatizer.lemmatize(w, pos='v') for w in words if w not in stop_words] # Lemmatization
    return " ".join(words)

def build_model():
    model = Sequential()

    # Vectorization - Word Embedding
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    
    # model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    # -> Accuracy 79%, Validation 70% - Overfitting
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))) 

    model.add(Dense(3, activation='softmax')) # 3 neurons - negative, neutral, positive
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
model_path = r"sentiment_lstm_final_ntlk_split.keras"

if os.path.exists(model_path):
    choice = input("Saved model found. Load it and skip training? (y/n): ")
    if choice.lower() == 'y':
        final_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully! Skipping Steps 1-6.")
        # Future Works - Prediction
        exit()

# ==========================================
# STEP 1: Load the dataset
# ==========================================
file_path = r"TweetsWithLabels.csv"

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

# *** Removing Rare Words - Keep Tops 5000
MAX_NB_WORDS = 5000
# MAX_NB_WORDS = 10000

MAX_SEQUENCE_LENGTH = 100 # Default Tweets - 40-60 Words
EMBEDDING_DIM = 100

# *** Tokenization
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(df['clean_text'].values)
X = tokenizer.texts_to_sequences(df['clean_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# *** Encode Labels
label_encoder = LabelEncoder()
y_integer = label_encoder.fit_transform(df['sentiment'])
y_onehot = to_categorical(y_integer)

# ==========================================
# STEP 4: Train and Test LSTM w/ 80/20 Split
# ==========================================
print(">>> Splitting data 80/20...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_integer
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

model = build_model()

# *** Early Stopping Techniques
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=15,
                    batch_size=128, # Common Batch Value
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# ==========================================
# STEP 5: RESULTS SUMMARY
# ==========================================
print("\n" + "=" * 50)
print("80/20 TRAIN-TEST SPLIT RESULTS")
print("=" * 50)
print(f"  Accuracy:   {accuracy:.4f}")
print(f"  Loss:       {loss:.4f}")
print(f"  Error Rate: {1 - accuracy:.4f}")

# ZeroR Baseline
zeror_accuracy = df['sentiment'].value_counts().max() / len(df)
print(f"  ZeroR Baseline:  {zeror_accuracy:.4f}")
print(f"  Improvement over ZeroR: {accuracy - zeror_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
      target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))
print("=" * 50)

# ==========================================
# STEP 6: VISUALIZE TRAINING HISTORY
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# STEP 7: SAVE MODEL
# ==========================================
save_path = r"sentiment_lstm_final_ntlk_split.keras"
model.save(save_path)
print(f"Model saved to {save_path}")

# Save results
results = pd.DataFrame({
    'Metric': ['Accuracy', 'Loss', 'Error Rate', 'ZeroR Baseline'],
    'Value': [accuracy, loss, 1 - accuracy, zeror_accuracy]
})
results_path = r"split_results_ntlk_split.csv"
results.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")

# ==========================================
# RESULTS
# ==========================================

# Epoch 1/15
# 172/172 ━━━━━━━━━━━━━━━━━━━━ 35s 165ms/step - accuracy: 0.5699 - loss: 0.8944 - val_accuracy: 0.6921 - val_loss: 0.7160
# Epoch 2/15
# 172/172 ━━━━━━━━━━━━━━━━━━━━ 32s 189ms/step - accuracy: 0.7307 - loss: 0.6581 - val_accuracy: 0.7003 - val_loss: 0.7015
# Epoch 3/15
# 172/172 ━━━━━━━━━━━━━━━━━━━━ 34s 198ms/step - accuracy: 0.7626 - loss: 0.5887 - val_accuracy: 0.7003 - val_loss: 0.7129
# Epoch 4/15
# 172/172 ━━━━━━━━━━━━━━━━━━━━ 39s 188ms/step - accuracy: 0.7853 - loss: 0.5446 - val_accuracy: 0.6996 - val_loss: 0.7368
# 172/172 ━━━━━━━━━━━━━━━━━━━━ 6s 28ms/step

# ==================================================
# 80/20 TRAIN-TEST SPLIT RESULTS
# ==================================================
#   Accuracy:   0.7003
#   Loss:       0.7015
#   Error Rate: 0.2997
#   ZeroR Baseline:  0.4045
#   Improvement over ZeroR: 0.2958

# Classification Report:
#               precision    recall  f1-score   support

#     negative       0.68      0.72      0.70      1556
#      neutral       0.67      0.66      0.66      2223
#     positive       0.77      0.74      0.76      1717

#     accuracy                           0.70      5496
#    macro avg       0.70      0.71      0.70      5496
# weighted avg       0.70      0.70      0.70      5496

# Confusion Matrix:
# [[1114  378   64]
#  [ 445 1461  317]
#  [  89  354 1274]]
