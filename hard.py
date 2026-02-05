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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
file_path = r"C:\Users\User\Downloads\tweets_v8.csv"
print(">>> Loading dataset...")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found. Creating dummy data.")
    # Dummy data for demonstration if file is missing
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

# Generate labels using TextBlob (The "Ground Truth" for our LSTM)
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
y = to_categorical(y_integer) 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# STEP 4: BUILD LSTM MODEL
# ==========================================
print(">>> Building LSTM Model...")

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# ==========================================
# STEP 5: TRAIN MODEL
# ==========================================
print(">>> Training Model...")
history = model.fit(X_train, y_train, 
                    epochs=5, 
                    batch_size=128,
                    validation_data=(X_test, y_test),
                    verbose=1)

# ==========================================
# STEP 6: EVALUATE & VISUALIZE
# ==========================================
print("\n>>> Evaluating...")
accr = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accr[1]:0.3f}')

# Plotting Results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()

plt.show()
