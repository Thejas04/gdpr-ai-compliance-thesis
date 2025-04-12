import os
import json
import torch
import logging
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from evaluate import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate
from tqdm.auto import tqdm
from accelerate import Accelerator

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config_path = '../config/config.json'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded successfully.")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    raise

# Unpack config values
model_name = config.get('model_name', 'nlpaueb/legal-bert-base-uncased')
epochs = config.get('epochs', 5)
batch_size = config.get('batch_size', 16)
learning_rate = config.get('learning_rate', 3e-5)
logging_steps = config.get('logging_steps', 50)
lora_r = config.get('lora_r', 8)
output_dir = config.get('output_dir', '../output/')

# Logging configuration details
logger.info(f"Model name: {model_name}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")

# Load tokenizer and model
try:
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# LoRA Configuration
try:
    lora_config = LoraConfig(r=lora_r)
    model = get_peft_model(model, lora_config)
    logger.info("LoRA configuration applied successfully.")
except Exception as e:
    logger.error(f"Error applying LoRA configuration: {e}")
    raise

# Label encoding
LABEL_MAP = {
    "Fully GDPR Compliant": 0,
    "Partially Compliant": 1,
    "Fully GDPR Non-Compliant": 2
}

def encode_labels(label):
    encoded_label = LABEL_MAP.get(label)
    if encoded_label is None:
        logger.warning(f"Unknown label encountered: {label}")
        return -1
    return encoded_label

# Load data
def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        texts, labels = [], []
        for item in data:
            label = encode_labels(item['label'])
            if label != -1:
                texts.append(item['text'])
                labels.append(label)
        logger.info(f"Loaded {len(texts)} samples from {filepath}")
        return Dataset.from_dict({'text': texts, 'label': labels})
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

train_dataset = load_data('../datasets/training_data.json')
val_dataset = load_data('../datasets/validation_data.json')

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")
val_dataset = val_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.rename_column("label", "labels")
val_dataset.set_format("torch")

# Custom evaluation metrics
metric = load("accuracy")
def compute_metrics(p):
    logger.info("Entering compute_metrics function...")
    try:
        preds = np.argmax(p.predictions, axis=-1)
        accuracy = accuracy_score(p.label_ids, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
        metrics = {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1
        }
        logger.info(f"Metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error in compute_metrics: {e}")
        raise

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    lr_scheduler_type="linear",
    warmup_steps=500,
    evaluation_strategy="epoch",
    logging_dir='../logs/',
    logging_steps=logging_steps,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=3,
    metric_for_best_model='eval_f1',
    report_to="none",
    dataloader_num_workers=4,
    label_names=["labels"],
)

# Trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Fine-tune the model
try:
    logger.info("Starting fine-tuning...")
    logger.info(f"TrainingArguments: {training_args}")
    trainer.train()
    logger.info("Fine-tuning completed successfully.")
except Exception as e:
    logger.error(f"Error during fine-tuning: {e}")
    raise

# Save the model
try:
    logger.info("Saving the fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved successfully.")
except Exception as e:
    logger.error(f"Error saving model: {e}")
    raise
