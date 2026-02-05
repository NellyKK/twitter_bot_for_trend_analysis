import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter

# --- NEW IMPORTS FOR ML ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
file_path = r"C:\Users\User\Downloads\tweets_v8.csv"
print(">>> Loading dataset...")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found. Creating dummy data.")
    df = pd.DataFrame({'text': ["I love AI", "Bad news", "Neutral", "AI is great"] * 100})

# Limit rows for performance
df = df.head(60000)
print(f"Processing the first {len(df)} rows.\n")

# 2. Define Stop Words
stop_words = {
    "the", "i", "to", "a", "and", "of", "in", "is", "it", 
    "for", "you", "on", "this", "that", "my", "with", "are", 
    "be", "at", "me", "have", "so", "but", "was", "not", 
    "just", "im", "your", "like", "all", "do", "we", "can",
    "from", "about", "an", "or", "has", "what", "if", "up", "out",
    "amp", "don", "t"
}

# 3. Data Cleaning
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("squid game", "squidgame")
    text = text.replace("squidgamenetflix", "squidgame")
    text = re.sub(r'[^\x00-\x7f]', r'', text) 
    text = re.sub(r"http\S+", "", text)      
    text = re.sub(r"@\w+", "", text)          
    text = re.sub(r"#", "", text)             
    text = re.sub(r"[^A-Za-z\s]", "", text)   
    
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

print(">>> Cleaning tweets...")
df["clean_text"] = df["text"].apply(clean_tweet)

# 4. Sentiment Analysis (Labeling for ML)
def get_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

print(">>> Calculating sentiment labels...")
df["sentiment"] = df["clean_text"].apply(get_sentiment)

# ==========================================
# STEP 4: MACHINE LEARNING PIPELINE
# ==========================================

# A. Define X (Features) and y (Target)
X = df['clean_text']
y = df['sentiment']  # Matches the column created above

# B. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining on {len(X_train)} tweets, Testing on {len(X_test)} tweets.")

# C. Vectorization (TF-IDF)
print(">>> Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# D. Train the Model (Logistic Regression)
print(">>> Training the Machine Learning Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# E. Evaluate
print(">>> Predicting on Test Data...")
y_pred = model.predict(X_test_vec)

# Results
print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# STEP 5: VISUALIZATION
# ==========================================

# --- Plot 1: Top Words (Original Request) ---
def get_top_words(data_subset):
    text_blob = " ".join(data_subset["clean_text"])
    words = text_blob.split()
    return Counter(words).most_common(10)

top_positive = get_top_words(df[df["sentiment"] == "Positive"])
top_negative = get_top_words(df[df["sentiment"] == "Negative"])
top_neutral  = get_top_words(df[df["sentiment"] == "Neutral"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_words(top_list, ax, title, color):
    if not top_list: return
    words, counts = zip(*top_list)
    sns.barplot(x=list(words), y=list(counts), ax=ax, palette=color)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

plot_words(top_positive, axes[0], "Top Positive Words", "Greens_r")
plot_words(top_negative, axes[1], "Top Negative Words", "Reds_r")
plot_words(top_neutral, axes[2], "Top Neutral Words", "Blues_r")
plt.tight_layout()
plt.show()

# --- Plot 2: Confusion Matrix (ML Visualization) ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.title("Confusion Matrix: Actual vs Predicted", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label (from TextBlob)")
plt.show()