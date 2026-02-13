# ================================
# Logistic Regression Sentiment Analysis on Tweets
# ================================

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# ==========================================
# 1. Configuration
# ==========================================
DATASET_PATH = r'C:\Users\User\Downloads\tweets.csv' 

def load_data(path):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        print(f"ERROR: The file must have columns named 'text' and 'sentiment'. Found: {list(df.columns)}")
        return None
    print(f"Success! Loaded {len(df)} rows.")
    return df

# ==========================================
# 2. Preprocessing
# ==========================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Remove negation words from the "stopword list" so we keep them in the text
negation_words = {'not', 'no', 'nor', 'never', "don't", "aren't", 
                  "couldn't", "didn't", "doesn't", "dont", "hadn't", 
                  "hasn't", "haven't", "isn't", "wasn't", "weren't", 
                  "won't", "wouldn't"}
stop_words = stop_words - negation_words
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                         #lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)   # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)       # remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # remove special chars (numbers/punctuation)
    text = re.sub(r'[^\x00-\x7f]', r'', text)   # removes non-ASCII characters
    text = re.sub(r"#", "", text)               # remove hashtag symbol

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ==========================================
# 3. Model Training
# ==========================================
def train_model(df):
    print("Cleaning data...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    X = df['clean_text']
    y = df['sentiment']

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        
        # solver = 'lbfgs' to handle 3+ classes (Positive/Negative/Neutral)
        ('clf', LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000))
    ])

    print("Training Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test

# ==========================================
# 4. Evaluation
# ==========================================
def evaluate_model(model, X_test, y_test):
    # 1. Generate Predictions
    y_pred = model.predict(X_test)
    
    # --- Standard Reports ---
    print("\n" + "="*30)
    print("      IMPROVED LOGISTIC REGRESSION MODEL REPORT      ")
    print("="*30)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.2%}") 
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix')
    plt.show()
 
def visualize_sentiment_distribution(df):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='sentiment', data=df, palette='viridis')
    plt.title('Distribution of Sentiments', fontsize=14)
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    
    for container in ax.containers:
        ax.bar_label(container)
        
    plt.show()
# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    if df is not None:
         # Visualize the Class Count
        print("Visualizing data distribution...")
        visualize_sentiment_distribution(df)
        model, X_test, y_test = train_model(df)
        
        # Prints the Accuracy and Report
        evaluate_model(model, X_test, y_test)