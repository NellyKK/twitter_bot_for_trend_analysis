import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import unicodedata

# --- NEW IMPORTS FOR ML ---
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 1. Load the dataset
file_path = r"TweetsWithLabels.csv"
print(">>> Loading dataset...")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found. Creating dummy data.")
    df = pd.DataFrame({'text': ["I love AI", "Bad news", "Neutral", "AI is great"] * 100})


# 2. Dataset exploration
print(f"Dataset has {len(df)} rows.\n")
print(df.info())
print(df.describe())
print(df.head())

# Sentiment Categories
print(f'The tweets are classified in the following 3 categories:{df['sentiment'].unique()}\n')
print(f'Value counts for each sentiment:\n{df["sentiment"].value_counts()}\n')

# Check Null Values
print(f'Check null values:\n{df.isnull().sum()}\n')

# Check Duplicate Values
print(f'Check duplicate values: {df.duplicated().sum()}\n')


# 3. Define Stop Words
stop_words = {
    "the", "i", "to", "a", "and", "of", "in", "it", 
    "for", "you", "on", "this", "that", "my", "with", 
    "be", "at", "me", "have", "so", "but", "was", "not", 
    "just", "im", "your", "like", "all", "do", "we",
    "an", "or", "has", "what", "if", "up", "out",
    "amp", "don", "t"
}

# Limit rows for performance
#df = df.head(60000)
#print(f"Processing the first {len(df)} rows.\n")

# 4. Data Preprocessing

# Drop missing values
df.dropna(inplace=True)

# Drop 'textID' and 'selected_text' columns
df.drop(columns=['textID', 'selected_text'], inplace=True)

# Function to clean text in tweets
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text) # Drop digits
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8') # Handel non-ASCII characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r"@\w+", "", text) # Remove mentions     
    text = re.sub(r"#", "", text) # Remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text) # Remove any non letter characters
    text = text.strip()
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words] # Remove stopwords
    return " ".join(filtered_words)

# Cleaning text in tweets
print(">>> Cleaning tweets...")
df["clean_text"] = df["text"].apply(clean_tweet)

# Tokenizing text for Lemmatization
df['clean_text'] = df['clean_text'].apply(lambda text: word_tokenize(text))

# Function to apply Lemmatization and joining back the text
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word, pos='v') for word in text])

# Applying Lemmatization
print(">>> Applying Lemmatization...")
df['clean_text'] = df['clean_text'].apply(lambda text: lemmatize_words(text))
print(f'Preprocessed data example:\n {df[['text', 'clean_text']].head(20)}')

# ==========================================
# STEP 4: MACHINE LEARNING PIPELINE
# ==========================================

# A. Define X (Features) and y (Target)
X = df['clean_text']
y = df['sentiment']

# B. Split the dataset into training (80%) and testing (20%)
# Stratification to keep the proprotion of classes 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# C. Vectorization (TF-IDF)
print(">>> Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# D. Train the Model (SVM)
print(">>> Training the Machine Learning Model...")
svm_model = SVC(random_state=42)
svm_model.fit(X_train_vec, y_train)

# E. Evaluate
print(">>> Predicting on Test Data...")
y_pred = svm_model.predict(X_test_vec)

# Results
print("\n--- MODEL PERFORMANCE BEFORE HYPERPARAMETER TUNING ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Hyperparameter Tuning

print(">>> Hyperparameter Tuning...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

#grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train_vec, y_train)

# Display best parameters
#print('Best parameters: ', grid_search.best_params_)
print('Best parameters: ', {'C': 1, 'gamma': 'scale', 'kernel': 'linear'})
# Best parameters:  {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}

# Train the best model
print(">>> Training the Machine Learning Model with Best Parameters...")
#best_svm = grid_search.best_estimator_
#y_pred_best = best_svm.predict(X_test_vec)

best_svm = SVC(C=1, gamma='scale', kernel='linear')
best_svm.fit(X_train_vec, y_train)


print(">>> Predicting on Test Data with Best SVM...")
y_pred_best = best_svm.predict(X_test_vec)

# Final Accuracy after Hyperparameter Tuning
final_acc = accuracy_score(y_test, y_pred_best)
print("\n--- MODEL PERFORMANCE AFTER HYPERPARAMETER TUNING ---")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# ==========================================
# STEP 5: VISUALIZATION
# ==========================================

# --- Plot 1: Top Words Per Sentiment ---
def get_top_words(data_subset):
    text_blob = " ".join(data_subset["clean_text"])
    words = text_blob.split()
    return Counter(words).most_common(10)

top_positive = get_top_words(df[df["sentiment"] == "positive"])
top_negative = get_top_words(df[df["sentiment"] == "negative"])
top_neutral  = get_top_words(df[df["sentiment"] == "neutral"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_words(top_list, ax, title, color):
    if not top_list: return
    words, counts = zip(*top_list)
    sns.barplot(x=list(words), y=list(counts), ax=ax, hue= list(words), palette=color, legend=False)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

plot_words(top_positive, axes[0], "Top Positive Words", "Greens_r")
plot_words(top_negative, axes[1], "Top Negative Words", "Reds_r")
plot_words(top_neutral, axes[2], "Top Neutral Words", "Blues_r")
plt.tight_layout()
plt.show()

# --- Plot 2: Confusion Matrix (ML Visualization) ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best, labels=["negative", "neutral", "positive"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.title("Confusion Matrix: Actual vs Predicted for SVM Model", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

