import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud

# 1. Load the dataset
file_path = r"C:\Users\User\Downloads\tweets_v8.csv"
print(">>> Loading dataset...")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found. Creating dummy data.")
    df = pd.DataFrame({'text': ["I love AI", "Bad news", "Neutral", "AI is great"]})

# Limit rows
df = df.head(10000)
print(f"Processing the first {len(df)} rows.\n")

# 2. Define Stop Words (Moved to TOP so clean_tweet can use it)
stop_words = {
    "the", "i", "to", "a", "and", "of", "in", "is", "it", 
    "for", "you", "on", "this", "that", "my", "with", "are", 
    "be", "at", "me", "have", "so", "but", "was", "not", 
    "just", "im", "your", "like", "all", "do", "we", "can",
    "from", "about", "an", "or", "has", "what", "if", "up", "out",
    "amp", "don", "t"
}

# 3. Data Cleaning (Now removes stop words inside the function)
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # --- MERGE VARIATIONS ---
    text = text.replace("squid game", "squidgame")
    text = text.replace("squidgamenetflix", "squidgame")
    
    # --- REGEX CLEANING ---
    text = re.sub(r'[^\x00-\x7f]', r'', text) # Remove non-English chars
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#", "", text)             # remove hashtag symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)   # remove special chars (numbers/punctuation)

    # --- STOP WORD REMOVAL ---
    # Split text into a list of words
    words = text.split()
    # Keep only words that are NOT in the stop_words set
    filtered_words = [w for w in words if w not in stop_words]
    # Join them back into a string
    text = " ".join(filtered_words)
    
    return text

print(">>> Cleaning tweets (and removing stop words)...")
df["clean_text"] = df["text"].apply(clean_tweet)

print("First 5 cleaned tweets:")
print(df["clean_text"].head()) 

# 4. Sentiment Analysis (Now runs on text without stop words)
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

print(">>> Calculating sentiment...")
df["sentiment"] = df["clean_text"].apply(get_sentiment)

# 5. Helper Function to Get Top Words
# (We don't need to filter stop words here anymore, because clean_tweet already did it!)
def get_top_words(data_subset):
    text_blob = " ".join(data_subset["clean_text"])
    words = text_blob.split()
    return Counter(words).most_common(10)

# 6. Get Top 10 for EACH Sentiment
print(">>> Analyzing word frequencies per sentiment...")
top_positive = get_top_words(df[df["sentiment"] == "Positive"])
top_negative = get_top_words(df[df["sentiment"] == "Negative"])
top_neutral  = get_top_words(df[df["sentiment"] == "Neutral"])

# 7. Generate Text Report
report = f"""
\nFinal Report: Top 10 Words by Sentiment 📊
-------------------------------------------
Total Tweets: {len(df)}

🟢 POSITIVE Top 10:
{top_positive}

🔴 NEGATIVE Top 10:
{top_negative}

🔵 NEUTRAL Top 10:
{top_neutral}
"""
print(report)

# 8. Visualization
print(">>> Generating Graphs...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def plot_words(top_list, ax, title, color):
    if not top_list:
        return
    words, counts = zip(*top_list)
    sns.barplot(x=list(words), y=list(counts), ax=ax, palette=color)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

plot_words(top_positive, axes[0], "Top Positive Words", "Greens_r")
plot_words(top_negative, axes[1], "Top Negative Words", "Reds_r")
plot_words(top_neutral, axes[2], "Top Neutral Words", "Blues_r")

plt.tight_layout()
plt.show()