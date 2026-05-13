with open('model_training.ipynb', 'r') as f:
    content = f.read()
content = content.replace('"    tokenizer=tokenizer,\\n",', '"    processing_class=tokenizer,\\n",')
with open('model_training.ipynb', 'w') as f:
    f.write(content)
print("Fixed notebook.")
