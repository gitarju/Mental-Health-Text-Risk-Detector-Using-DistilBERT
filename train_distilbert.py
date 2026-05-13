import pandas as pd
import os
import glob
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

print("Loading data...")
raw_data_dir = 'Dataset/Original Reddit Data/raw data/'
all_files = glob.glob(os.path.join(raw_data_dir, '**', '*.csv'), recursive=True)
dfs = []
for f in all_files:
    df = pd.read_csv(f)
    if 'selftext' not in df.columns:
        continue
    
    fname = os.path.basename(f).lower()
    if 'dep' in fname:
        label = 'Depression'
    elif 'ani' in fname or 'anx' in fname:
        label = 'Anxiety'
    elif 'lone' in fname:
        label = 'Loneliness'
    elif 'mh' in fname:
        label = 'Mental Health'
    elif 'sw' in fname or 'suicide' in fname:
        label = 'High Risk (SW)'
    else:
        label = 'Other'
        
    df['Label'] = label
    # Sample just 100 rows per file for fast local execution
    df_sampled = df.dropna(subset=['selftext']).sample(min(100, len(df.dropna(subset=['selftext']))), random_state=42)
    dfs.append(df_sampled[['selftext', 'Label']])

train_df = pd.concat(dfs, ignore_index=True)
train_df['cleaned_text'] = train_df['selftext'].apply(clean_text)
train_df = train_df[train_df['cleaned_text'].str.strip() != ""]

print(f"Training data size (abbreviated): {len(train_df)}")

# Define Labels
labels = train_df['Label'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

train_df['label_id'] = train_df['Label'].map(label2id)

print(f"Labels: {label2id}")

# Load Tokenizer
print("Loading Tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create Dataset
hf_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label_id']])
# Handle '__index_level_0__' column that might appear from pandas
if '__index_level_0__' in hf_dataset.column_names:
    hf_dataset = hf_dataset.remove_columns(['__index_level_0__'])
hf_dataset = hf_dataset.rename_column("label_id", "label")
hf_dataset = hf_dataset.rename_column("cleaned_text", "text")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing data...")
tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

# Split dataset
train_testvalid = tokenized_datasets.train_test_split(test_size=0.1)

# Load Model
print("Loading Model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(labels), 
    id2label=id2label, 
    label2id=label2id
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./distilbert_results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1, # Very fast training for demonstration
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_testvalid["train"],
    eval_dataset=train_testvalid["test"],
    processing_class=tokenizer,
)

print("Starting Training...")
trainer.train()

# Save Model
output_dir = "models/distilbert"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
