import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud  # <--- New Import

# 1. Load the dataset
file_path = r"C:\Users\User\Downloads\tweets_v8.csv"
print(">>> Loading dataset...")

try:
    df = pd.read_csv(file_path,encoding='utf-8')
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: File not found. Creating dummy data.")
    df = pd.DataFrame({'text': ["I love AI", "Bad news", "Neutral", "AI is great"]})

# --- SPEED FIX: Limit to first 5000 rows ---
# df = df.head(5000)
df = df.head(10000)
print(f"Processing the first {len(df)} rows to save time.\n")

# 2. Data Cleaning
# def clean_tweet(text):
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r"http\S+", "", text)       # remove URLs
#     text = re.sub(r"@\w+", "", text)          # remove mentions
#     text = re.sub(r"#", "", text)             # remove hashtags
#     text = re.sub(r"[^A-Za-z\s]", "", text)   # remove special chars
#     return text.lower()

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower() # Make everything lowercase first
    
    # --- FIX: MERGE VARIATIONS ---
    text = text.replace("squid game", "squidgame")       # Turn "squid game" into "squidgame"
    # text = text.replace("squidgamenetflix", "squidgame") # Fix the netflix hashtag
    # -----------------------------
    # text = text.strip()
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#", "", text)             # remove hashtag symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)   # remove special chars
    return text

print(">>> Cleaning tweets...")
df["clean_text"] = df["text"].apply(clean_tweet)

print(df.head())
stop_words = {
    "the", "i", "to", "a", "and", "of", "in", "is", "it", 
    "for", "you", "on", "this", "that", "my", "with", "are", 
    "be", "at", "me", "have", "so", "but", "was", "not", 
    "just", "im", "your", "like", "all", "do", "we", "can",
    "from", "about", "an", "or", "has", "what", "if", "up", "out",
    "amp", "don", "t"
}


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

print(">>> Calculating sentiment...")
df["sentiment"] = df["clean_text"].apply(get_sentiment)

# # 4. Helper Function to Get Top Words
# stop_words = {
#     "the", "i", "to", "a", "and", "of", "in", "is", "it", 
#     "for", "you", "on", "this", "that", "my", "with", "are", 
#     "be", "at", "me", "have", "so", "but", "was", "not", 
#     "just", "im", "your", "like", "all", "do", "we", "can",
#     "from", "about", "an", "or", "has", "what", "if", "up", "out",
#     "amp", "don", "t"
# }

def get_top_words(data_subset):
    text_blob = " ".join(data_subset["clean_text"])
    words = text_blob.split()
    # Filter stop words
    meaningful_words = [word for word in words if word not in stop_words]
    # Count and return top 10
    return Counter(meaningful_words).most_common(10)

# 5. Get Top 10 for EACH Sentiment
print(">>> Analyzing word frequencies per sentiment...")
top_positive = get_top_words(df[df["sentiment"] == "Positive"])
top_negative = get_top_words(df[df["sentiment"] == "Negative"])
top_neutral  = get_top_words(df[df["sentiment"] == "Neutral"])

# 6. Generate Text Report
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

# 7. Visualization (3 Bar Charts Side-by-Side)
print(">>> Generating Graphs...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Helper to plot on a specific axis
def plot_words(top_list, ax, title, color):
    if not top_list: # Handle empty data
        return
    words, counts = zip(*top_list)
    sns.barplot(x=list(words), y=list(counts), ax=ax, palette=color)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

# Plot Positive
plot_words(top_positive, axes[0], "Top Positive Words", "Greens_r")

# Plot Negative
plot_words(top_negative, axes[1], "Top Negative Words", "Reds_r")

# Plot Neutral
plot_words(top_neutral, axes[2], "Top Neutral Words", "Blues_r")

plt.tight_layout()
plt.show()