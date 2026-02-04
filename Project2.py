import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter
# import tweepy

# 1. Load the dataset

#TWITTER
# client = tweepy.Client(
#     bearer_token="YOUR_BEARER_TOKEN",
#     consumer_key="API_KEY",
#     consumer_secret="API_SECRET",
#     access_token="ACCESS_TOKEN",
#     access_token_secret="ACCESS_TOKEN_SECRET"
# )

#KAGGLE
file_path = r"C:\Users\User\Downloads\tweets_v8.csv"
print(">>> Loading dataset...")

try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found. Creating dummy data.")
    df = pd.DataFrame({'text': ["I love AI", "Bad news", "Neutral", "AI is great"]})



#2. DATA EXPLORATION
print("Columns: ",df.columns)
# Columns:  Index(['user_name', 'user_location', 'user_description', 'user_created',
#        'user_followers', 'user_friends', 'user_favourites', 'user_verified',
#        'date', 'text', 'source', 'is_retweet'],     

nullValues = df.isnull().sum()
print(f'Null values: \n{nullValues}')

# --- SPEED FIX: Limit to first 5000 rows for testing ---
# (Comment this line out later if you want to analyze ALL 80,000 tweets)
# df = df.head(5000)
print(f"Processing the first {len(df)} rows to save time.\n")
# Null values: 
# user_name               4
# user_location       23870
# user_description     5211
# user_created            0
# user_followers          0
# user_friends            0
# user_favourites         0
# user_verified           0
# date                    0
# text                    0
# source                  0
# is_retweet              0
# dtype: int64

#Selection attribute : text
df = df[['text']]

# 2. Data Cleaning
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#", "", text)             # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)   # remove special chars
    return text.lower()

print(">>> Cleaning tweets... (Please wait)")
df["clean_text"] = df["text"].apply(clean_tweet)

# 3. Sentiment Analysis
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

print(">>> Calculating sentiment... (This takes the longest)")
df["sentiment"] = df["clean_text"].apply(get_sentiment)

# 4. Generate Report
sentiment_counts = df["sentiment"].value_counts()
all_words = " ".join(df["clean_text"]).split()
common_words = Counter(all_words).most_common(5)

report = f"""
\nFinal Report 📊
-----------------------
Total Tweets Analyzed: {len(df)}

Sentiment Breakdown:
Positive: {sentiment_counts.get('Positive', 0)}
Negative: {sentiment_counts.get('Negative', 0)}
Neutral:  {sentiment_counts.get('Neutral', 0)}

Top 5 Common Words:
{common_words}
"""
print(report)

# 5. Visualization
print(">>> Generating Graph...")
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.show()