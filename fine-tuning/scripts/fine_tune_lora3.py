import os
import json
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score
import evaluate
from tqdm.auto import tqdm
from accelerate import Accelerator

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config_path = '../config/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

model_name = config.get('model_name', 'nlpaueb/legal-bert-base-uncased')
epochs = config.get('epochs', 5)
batch_size = config.get('batch_size', 16)
learning_rate = config.get('learning_rate', 3e-5)
logging_steps = config.get('logging_steps', 50)
lora_r = config.get('lora_r', 8)
gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"../output/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=9)

# Ensure gradients are enabled for model parameters
for param in model.parameters():
    param.requires_grad = True
logger.info("All model parameters set to require gradients.")

# LoRA configuration
lora_config = LoraConfig(r=lora_r)
model = get_peft_model(model, lora_config)

# Label mapping
LABEL_MAP = {
    "Fully GDPR Compliant": 0,
    "Fully GDPR Non-Compliant": 1,
    "Partially Compliant (Article 4(11) Compliant, Article 7 Not)": 2,
    "Partially Compliant (Article 4(11) Non-Compliant, Article 7 Compliant)": 3,
    "Partially Compliant (Article 4(11) & Article 7 Partial Violation)": 4,
    "Partially Compliant (Granular Consent Missing, Withdrawal Available)": 5,
    "Partially Compliant (No Clear Consent, But Withdrawal Available)": 6,
    "Partially Compliant (Consent Given but Consent Process Is Difficult)": 7,
    "Partially Compliant (Consent Issues and Transparency Issues)": 8
}

# Encode labels
def encode_labels(label):
    return LABEL_MAP.get(label, -1)

# Load data
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data:
        label = encode_labels(item['label'])
        if label != -1:
            texts.append(item['text'])
            labels.append(label)
    return Dataset.from_dict({'text': texts, 'labels': labels})

train_dataset = load_data('../datasets/synthetic_training_data.json')
val_dataset = load_data('../datasets/synthetic_validation_data.json')

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(p.label_ids, preds)
    kappa = cohen_kappa_score(p.label_ids, preds)
    cm = confusion_matrix(p.label_ids, preds)
    report = classification_report(p.label_ids, preds, target_names=LABEL_MAP.keys())
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")
    return {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'mcc': mcc,
        'kappa': kappa
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    eval_strategy="epoch",
    logging_steps=logging_steps,
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    save_total_limit=3,
    fp16=True,
    optim="adamw_torch",
    label_names=["labels"],
    gradient_checkpointing=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Training
trainer.train()
