import os
import json
import logging
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load validation data
logger.info("Loading validation data...")
validation_data_path = '../datasets/validation_data.json'
cleaned_validation_data_path = '../datasets/cleaned_validation_data.json'

try:
    with open(validation_data_path, 'r') as f:
        validation_data = json.load(f)
    logger.info(f"Loaded {len(validation_data)} validation samples.")
except Exception as e:
    logger.error(f"Error loading validation data: {e}")
    raise

# Clean validation data
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    return ' '.join(text.split())

cleaned_validation_data = []
logger.info("Cleaning validation data...")
for item in validation_data:
    cleaned_text = clean_text(item['text'])
    cleaned_item = {'text': cleaned_text, 'label': item['label']}
    cleaned_validation_data.append(cleaned_item)

logger.info(f"Cleaned validation data: {len(cleaned_validation_data)} samples.")

# Save cleaned validation data
try:
    with open(cleaned_validation_data_path, 'w') as f:
        json.dump(cleaned_validation_data, f, indent=4)
    logger.info("Cleaned validation data saved successfully.")
except Exception as e:
    logger.error(f"Error saving cleaned validation data: {e}")
    raise
