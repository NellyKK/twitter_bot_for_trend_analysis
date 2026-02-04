# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter

# Note: These are imported but not currently used in the script below
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load the dataset
# Make sure the path is correct. Using r"" handles backslashes in Windows paths.
file_path = r"C:\Users\User\Downloads\tweets_v8.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
    # Create dummy data so the code runs for demonstration if file is missing
    data = {'text': ["I love AI! #tech", "This is bad news :(", "Neutral tweet here", "AI is amazing", "Python is great"]}
    df = pd.DataFrame(data)

# 3. Data Inspection
print("\n--- Info & Missing Data ---")
df.info()
print("\nMissing values:\n", df.isnull().sum())

# 4. Data Cleaning
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#", "", text)             # remove hashtag symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)   # remove special chars
    return text.lower()

# Apply cleaning (This creates the 'clean_text' column needed later)
if 'text' in df.columns:
    df["clean_text"] = df["text"].apply(clean_tweet)
else:
    print("Error: Column 'text' not found in dataset. Please check column names.")

# 5. Sentiment Analysis
def get_sentiment(text):
    if not isinstance(text, str):
        return "Neutral"