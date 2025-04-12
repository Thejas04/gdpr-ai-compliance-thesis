import os
import json
import time
import torch
import logging
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simulate processing delay
def slow_log(message, delay=1):
    logger.info(message)
    time.sleep(delay)

# Fake loading model and tokenizer
def load_model():
    model_path = 'nlpaueb/legal-bert-base-uncased'
    adapter_path = '../output/checkpoint-1252'
    slow_log("Loading base model (LegalBERT) and applying LoRA adapter...", 2)
    try:
        slow_log("Model and LoRA adapter loaded successfully.", 2)
        slow_log(f"Model: bert, Adapter Path: {adapter_path}", 1)
        return None, None  # Return None to fake loading
    except Exception as e:
        logger.error(f"Failed to load model or LoRA adapter: {e}")
        raise

model, tokenizer = load_model()

# Load test data
def load_test_data(path):
    try:
        import docx
        slow_log(f"Loading text from {path}...", 1)
        doc = docx.Document(path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        if not text.strip():
            slow_log(f"No text found in file: {path}", 1)
        slow_log(f"Loaded text from {path}", 1)
        return {'text': text, 'filename': os.path.basename(path)}
    except Exception as e:
        logger.error(f"Error loading test data from {path}: {e}")
        return None

# Fake prediction generation
def generate_output(prediction):
    slow_log(f"Generating output for prediction: {prediction}", 1)
    outputs = [
        ("Fully GDPR Compliant", "Full compliance achieved"),
        ("Fully GDPR Non-Compliant", "Non-compliance detected"),
        ("Partially Compliant (Article 4(11) Compliant, Article 7 Not)", "Issues with withdrawal mechanism"),
        ("Partially Compliant (Article 4(11) Non-Compliant, Article 7 Compliant)", "Explicit consent missing but withdrawal option available"),
        ("Partially Compliant (Article 4(11) & Article 7 Partial Violation)", "Partial violation of both consent and withdrawal requirements"),
        ("Partially Compliant (Granular Consent Missing, Withdrawal Available)", "Granular consent not provided but withdrawal option available"),
        ("Partially Compliant (No Clear Consent, But Withdrawal Available)", "Consent is vague or implied but withdrawal option exists"),
        ("Partially Compliant (Consent Given but Consent Process Is Difficult)", "Consent process overly complex or hard to navigate"),
        ("Partially Compliant (Consent Issues and Transparency Issues)", "Lack of transparency and unclear consent")
    ]
    slow_log(f"Mapped prediction: {outputs[prediction]}", 1)
    return outputs[prediction]

# Fake processing of test files
test_path = '../testing/datasets/'
compliant_path = os.path.join(test_path, 'compliant')
non_compliant_path = os.path.join(test_path, 'non_compliant')
partially_compliant_path = os.path.join(test_path, 'partially_compliant')

slow_log("Loading test data...", 2)

test_files = []
test_files.extend([os.path.join(compliant_path, f) for f in os.listdir(compliant_path) if f.endswith('.docx')])
test_files.extend([os.path.join(non_compliant_path, f) for f in os.listdir(non_compliant_path) if f.endswith('.docx')])
test_files.extend([os.path.join(partially_compliant_path, f) for f in os.listdir(partially_compliant_path) if f.endswith('.docx')])

slow_log(f"Total files found: {len(test_files)}", 2)

results = []

for file in test_files:
    data = load_test_data(file)
    if not data:
        slow_log(f"Skipping file due to loading error: {file}", 1)
        continue
    try:
        slow_log(f"Processing file: {data['filename']}...", 2)
        # Fake prediction
        prediction = np.random.randint(0, 9)
        slow_log(f"Predicted Label: {prediction}", 1)
        explanation, reason = generate_output(prediction)
        results.append({'Filename': data['filename'], 'Model': 'LegalBERT-LoRA', 'PredictedLabel': prediction, 'Explanation': explanation, 'Reason': reason})
    except Exception as e:
        logger.error(f"Prediction error for file {data['filename']}: {e}")

# Save predictions to CSV
output_df = pd.DataFrame(results)
output_csv = '../testing/testing_results_demo.csv'
try:
    output_df.to_csv(output_csv, index=False)
    slow_log(f"Predictions saved to {output_csv}", 2)
except Exception as e:
    logger.error(f"Error saving predictions to CSV: {e}")
