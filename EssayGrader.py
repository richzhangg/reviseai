from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate


dataset = load_dataset("csv", data_files="essaygrader_data.csv")


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-base",
    num_labels=11  # scores 0–10 = 11 classes
)


def preprocess(examples):
    return tokenizer(examples["Essay"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(preprocess, batched=True)


if "train" not in tokenized_dataset:
    tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

# Load shit
accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

# setup training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# train it
trainer.train()

# save it
trainer.save_model("./essay_scoring_model")

print("yayaya passed")
