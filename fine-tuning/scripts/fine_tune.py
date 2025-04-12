import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from tqdm import tqdm

# Load configuration
with open(os.path.expanduser("~/fine-tuning/config/config.json"), "r") as file:
    config = json.load(file)

accelerator = Accelerator()

# Load model and tokenizer
model_name = config["model_name"]
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels now
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA for fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"]
)
model = get_peft_model(model, lora_config)

# Load datasets
train_dataset = load_dataset("json", data_files=config["train_file"])["train"]
val_dataset = load_dataset("json", data_files=config["validation_file"])["train"]

# Tokenization
def tokenize_function(examples):
    texts = []
    for i in range(len(examples["text"])):
        classification = examples["label"][i]  # Updated to match the new dataset format
        explanation = examples["structured_explanation"][i]
        reason = examples["reason"][i]

        # Construct the text input for the model
        text = (
            f"Classification: {classification}. "
            f"Reason: {reason}. "
            f"Structured Explanation: {explanation}."
        )
        texts.append(text)

    # Tokenize the entire batch of texts at once
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    # Robust label mapping
    label_map = {
        "Fully GDPR Compliant": 0,
        "Fully GDPR Non-Compliant": 1,
        "Partially Compliant": 2  # All subcategories of partial compliance will fall under this
    }

    labels = []
    for i in range(len(examples["label"])):
        label = label_map.get(examples["label"][i], -1)
        if label == -1:
            print(f"⚠️ Warning: Unrecognized classification '{examples['label'][i]}' - using default label 2.")
            label = 2  # Default to "Partially Compliant"
        labels.append(label)
    encodings["labels"] = labels

    return encodings

# Tokenize datasets
print("🚀 Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
print("✅ Tokenization complete!")

# Training arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    evaluation_strategy=config["evaluation_strategy"],
    eval_steps=config["eval_steps"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_eval_batch_size"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    warmup_steps=config["warmup_steps"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    logging_dir=config["logging_dir"],
    logging_steps=config["logging_steps"],
    save_steps=config["save_steps"],
    fp16=config["fp16"],
    seed=config["seed"],
    report_to="none"  # Disable reporting to avoid unnecessary logs
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Fine-Tuning
print("🚀 Starting fine-tuning...")
try:
    trainer.train()
    trainer.save_model(config["output_dir"])
    print("✅ Fine-tuning complete!")
except Exception as e:
    print(f"❌ An error occurred during fine-tuning: {str(e)}")
