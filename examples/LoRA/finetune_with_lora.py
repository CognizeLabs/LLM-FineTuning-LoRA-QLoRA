from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from transformers.adapters import LoRAConfig

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load data (for example purposes, replace with your dataset)
train_texts = ["I love this!", "I hate this!"]
train_labels = [1, 0]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")

# Prepare the adapter configuration
lora_config = LoRAConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"]
)
model.add_adapter("lora_adapter", config=lora_config)
model.train_adapter("lora_adapter")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
)

trainer.train()
