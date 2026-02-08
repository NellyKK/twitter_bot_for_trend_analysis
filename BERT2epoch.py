#BERT 2 epoch  run on gg colab

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import torch

# NLP Libraries
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Transformers / Deep Learning
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Download resources once
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ==========================================
# 1. Data Loading
# ==========================================
print(">>> Initializing Data Loading...")
df = None


try:
    dataset_path = kagglehub.dataset_download("yasserh/twitter-tweets-sentiment-dataset")
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in directory.")

    csv_path = os.path.join(dataset_path, csv_files[0])
    print(f">>> Loading dataset from: {csv_path}")
    #read dataset
    df = pd.read_csv(csv_path)

    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError(f"Missing columns. Required: ['text', 'sentiment']")

    print(f"    Loaded {len(df)} rows.")

except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# ==========================================
# 2. Data Cleaning
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()     #lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # remove URLs
    text = re.sub(r'@\w+', '', text)                    # remove mentions
    text = re.sub(r'#', '', text)                       # remove hashtag symbol
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    # NOTE: We are NOT removing stopwords, punctuation, or emojis.
    # We are NOT lemmatizing.
def preprocess_data(df):
    print(">>> Preprocessing data...")
    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)
    df['label'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    return df
# ==========================================
# 3. Model Training Pipeline
# ==========================================
def train_bert_model(df):
    print(">>> Preparing Training & Test sets...")

     # Split Data
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42 )
    print(f"\nTraining on {len(train_df)} tweets, Testing on {len( test_df)} tweets.")

    # Initialize Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(
            examples['clean_text'],
            padding="max_length",
            truncation=True,
            max_length=128 #maxlength of tweet is 280
        )

    # Convert to Datasets
    train_ds = Dataset.from_pandas(train_df[['clean_text', 'label']])
    test_ds = Dataset.from_pandas(test_df[['clean_text', 'label']])

    # Apply Tokenization
    print("    Tokenizing data...")
    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    # Set format for PyTorch
    cols = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format("torch", columns=cols)
    test_ds.set_format("torch", columns=cols)

    # Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Initializing BERT on {device}...")

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3).to(device)

    # Training Arguments
    args = TrainingArguments(
        output_dir="./bert_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=200,
        report_to="none"
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds, # Changed from test_df to test_ds
        compute_metrics=compute_metrics
    )

    print(">>> Starting Training...")
    trainer.train()

    return trainer, tokenizer, test_df


# ==========================================
# 4. Evaluation
# ==========================================
def evaluate_model(trainer, test_df):
    print("\n>>> Running Evaluation...")

    preds_output = trainer.predict(trainer.eval_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = test_df['label'].values

    y_pred_label = [{0: 'negative', 1: 'neutral', 2: 'positive'}[i] for i in y_pred]
    y_true_label = [{0: 'negative', 1: 'neutral', 2: 'positive'}[i] for i in y_true]

    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true_label, y_pred_label))

    # Show Samples
    print("\n>>> Sample 30 Predictions:")
    results = pd.DataFrame({
        'Text': test_df['text'].values[:30],
        'Actual': y_true_label[:30],
        'Predicted': y_pred_label[:30]})
    print(results.to_string(index=False))

    # Confusion Matrix
    cm = confusion_matrix(y_true_label, y_pred_label, labels=['negative', 'neutral', 'positive'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Poitives'],
                yticklabels=['Negative', 'Neutral', 'Poitives'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()

    # --- ERROR ANALYSIS START ---
    print("\n>>> ERROR ANALYSIS: Inspecting Mistakes...")

    # Create a DataFrame containing Text, Actual, and Predicted
    analysis_df = pd.DataFrame({
        'Text': test_df['text'].values,
        'Actual_ID': y_true,          # 0, 1, 2
        'Predicted_ID': y_pred,       # 0, 1, 2
        'Actual_Label': y_true_label,
        'Predicted_Label': y_pred_label
    })

    # 1. Filter: Actual is Negative (0), but Predicted Positive (2)
    # These are critical errors (Model thinks angry customer is happy)
    false_positives = analysis_df[
        (analysis_df['Actual_ID'] == 0) & (analysis_df['Predicted_ID'] == 2)
    ]

    print(f"\n[!] FALSE POSITIVES (Actual: Negative -> Predicted: Positive)")
    print(f"    Found {len(false_positives)} mistakes.")
    if len(false_positives) > 0:
        print(false_positives[['Text', 'Actual_Label', 'Predicted_Label']].head(50).to_string(index=False))
    else:
        print("    Great! No False Positives found.")

    # 2. Filter: Actual is Positive (2), but Predicted Negative (0)
    # These are missed opportunities (Model thinks happy customer is angry)
    false_negatives = analysis_df[
        (analysis_df['Actual_ID'] == 2) & (analysis_df['Predicted_ID'] == 0)
    ]

    print(f"\n[!] FALSE NEGATIVES (Actual: Positive -> Predicted: Negative)")
    print(f"    Found {len(false_negatives)} mistakes.")
    if len(false_negatives) > 0:
        print(false_negatives[['Text', 'Actual_Label', 'Predicted_Label']].head(50).to_string(index=False))
    else:
        print("    Great! No False Negatives found.")

    # --- ERROR ANALYSIS END ---


def predict_custom(trainer, tokenizer, texts):
    print("\n>>> Custom Test Predictions:")
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    device = trainer.model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = trainer.model(**inputs)

    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    for text, pred in zip(texts, preds):
        print(f"Text: '{text}'  ->  Sentiment: {id2label[pred]}")
# ==========================================
# 5. Visualization Functions
# ==========================================
def visualize_sentiment_distribution(df):
    """Bar chart of the class distribution"""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='sentiment', data=df, palette='viridis', hue='sentiment', legend=False)
    plt.title('Distribution of Original Sentiments', fontsize=14)
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container)
    plt.show()

def plot_training_history(trainer):
    """Plots Loss and Accuracy curves from HuggingFace Trainer logs"""
    print("\n>>> Plotting Training History...")

    # 1. Extract logs
    history = trainer.state.log_history
    history_df = pd.DataFrame(history)

    # 2. Separate Training and Validation logs
    # Training logs have 'loss', Validation logs have 'eval_loss'
    train_logs = history_df[history_df['loss'].notna()]
    val_logs = history_df[history_df['eval_loss'].notna()]

    plt.figure(figsize=(12, 5))

    # --- Plot 1: LOSS ---
    plt.subplot(1, 2, 1)
    plt.title('Loss Over Time')
    plt.plot(train_logs['epoch'], train_logs['loss'], label='Training Loss')
    plt.plot(val_logs['epoch'], val_logs['eval_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: ACCURACY ---
    plt.subplot(1, 2, 2)
    plt.title('Validation Accuracy')
    if 'eval_accuracy' in val_logs.columns:
        plt.plot(val_logs['epoch'], val_logs['eval_accuracy'], label='Val Accuracy', color='orange', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    if df is not None:
        # 3. Preprocess
        df = preprocess_data(df)

        # 4. Visualize Data Distribution (Requested Function 1)
        visualize_sentiment_distribution(df)

        # 5. Train Model
        trainer, tokenizer, test_df = train_bert_model(df)

        # 6. Visualize Training History (Requested Function 2)
        plot_training_history(trainer)

        # 7. Evaluate
        evaluate_model(trainer, test_df)

        # 8. Custom Test
        custom_tweets = [
            "This is absolutely amazing!",
            "I am not happy with this service.",
            "It is what it is, I guess.",
            "Don't worry, it's fine."
        ]
        predict_custom(trainer, tokenizer, custom_tweets)
