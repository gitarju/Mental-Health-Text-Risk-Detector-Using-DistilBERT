import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

output_dir = "models/distilbert"
os.makedirs(output_dir, exist_ok=True)

print("Downloading base model...")
model_name = "distilbert-base-uncased"

# Set up labels
labels = ['Depression', 'Anxiety', 'Loneliness', 'Mental Health', 'High Risk (SW)', 'Other']
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Base model saved to {output_dir}")
