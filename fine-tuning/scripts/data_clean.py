import os
import json
import logging
import random
from transformers import pipeline
from tqdm import tqdm
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
dataset_path = '../datasets/training_data.json'
cleaned_dataset_path = '../datasets/cleaned_training_data.json'
augmented_dataset_path = '../datasets/augmented_training_data.json'

# Load data
def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Save data
def save_data(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Data cleaning
def clean_data(data):
    cleaned_data = []
    seen_texts = set()
    for item in data:
        text = item['text'].strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            cleaned_data.append(item)
    return cleaned_data

# Data augmentation with paraphrasing
def augment_data(data):
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    augmented_data = []
    for item in tqdm(data, desc="Augmenting Data"):
        paraphrased = paraphraser(item['text'])[0]['generated_text']
        augmented_item = {
            "text": paraphrased,
            "label": item['label']
        }
        augmented_data.append(augmented_item)
    return augmented_data

# Clean data
logger.info("Loading data...")
raw_data = load_data(dataset_path)
logger.info(f"Loaded {len(raw_data)} samples.")

logger.info("Cleaning data...")
cleaned_data = clean_data(raw_data)
logger.info(f"Cleaned data: {len(cleaned_data)} samples.")
save_data(cleaned_dataset_path, cleaned_data)

# Augment data
logger.info("Augmenting data...")
augmented_data = augment_data(cleaned_data)
combined_data = cleaned_data + augmented_data
logger.info(f"Total data after augmentation: {len(combined_data)} samples.")
save_data(augmented_dataset_path, combined_data)

logger.info("Data cleaning and augmentation completed successfully.")
