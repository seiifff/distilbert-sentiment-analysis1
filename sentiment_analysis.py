from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import torch

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Load custom CSV dataset
df = pd.read_csv("train.csv")

# Encode labels
label_map = {"positive": 1, "negative": 0}
df["label"] = df["sentiment"].map(label_map)

# Convert to HuggingFace dataset
from datasets import Dataset
dataset = Dataset.from_pandas(df)

# Tokenize
def tokenize_function(example):
    return tokenizer(example["review"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_ds = tokenized_dataset["train"]
test_ds = tokenized_dataset["test"]

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# Save model
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

# Inference function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    return "Positive" if prediction == 1 else "Negative"

# Example usage
print(predict_sentiment("This movie was really enjoyable and well-acted."))