import os
import json
import torch
import logging
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the fine-tuned model and tokenizer
def load_model():
    model_path = 'nlpaueb/legal-bert-base-uncased'
    adapter_path = '../output/checkpoint-1252'
    logger.info("Loading base model (LegalBERT) and applying LoRA adapter...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model and LoRA adapter loaded successfully.")
        logger.info(f"Model: {model.config.model_type}, Adapter Path: {adapter_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model or LoRA adapter: {e}")
        raise

model, tokenizer = load_model()

# Load test data
def load_test_data(path):
    try:
        import docx
        doc = docx.Document(path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        if not text.strip():
            logger.warning(f"No text found in file: {path}")
        logger.info(f"Loaded text from {path}")
        return {'text': text, 'filename': os.path.basename(path)}
    except Exception as e:
        logger.error(f"Error loading test data from {path}: {e}")
        return None

# Output generation based on prediction
def generate_output(prediction):
    logger.info(f"Generating output for prediction: {prediction}")
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
    if 0 <= prediction < len(outputs):
        logger.info(f"Mapped prediction: {outputs[prediction]}")
        return outputs[prediction]
    return "Unknown Prediction", "Unexpected output from model"

# Load test files
test_path = '../testing/datasets/'

compliant_path = os.path.join(test_path, 'compliant')
non_compliant_path = os.path.join(test_path, 'non_compliant')
partially_compliant_path = os.path.join(test_path, 'partially_compliant')

logger.info("Loading test data...")

test_files = []
test_files.extend([os.path.join(compliant_path, f) for f in os.listdir(compliant_path) if f.endswith('.docx')])
test_files.extend([os.path.join(non_compliant_path, f) for f in os.listdir(non_compliant_path) if f.endswith('.docx')])
test_files.extend([os.path.join(partially_compliant_path, f) for f in os.listdir(partially_compliant_path) if f.endswith('.docx')])

logger.info(f"Total files found: {len(test_files)}")

# Generate predictions and save to CSV
results = []

for file in test_files:
    data = load_test_data(file)
    if not data:
        logger.warning(f"Skipping file due to loading error: {file}")
        continue
    try:
        with torch.no_grad():
            inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            logger.info(f"Logits: {logits}")
            prediction = np.argmax(logits, axis=-1)[0]
            logger.info(f"Predicted Label: {prediction}")
            prediction = torch.argmax(outputs.logits, dim=1).item()
        explanation, reason = generate_output(prediction)
        prompt_snippet = data['text'][:500] + '...' if len(data['text']) > 500 else data['text']
        results.append({
            'Filename': data['filename'],
            'Model': 'LegalBERT-LoRA',
            'Prompt': prompt_snippet,
            'PredictedLabel': prediction,
            'Explanation': explanation,
            'Reason': reason
        })
        logger.info(f"Prediction for {data['filename']}: {explanation} - {reason}")
    except Exception as e:
        logger.error(f"Prediction error for file {data['filename']}: {e}")

# Save predictions to CSV
output_df = pd.DataFrame(results)
output_csv = '../testing/testing_results.csv'
try:
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to {output_csv}")
    logger.info("Example predictions:")
    logger.info("Confusion Matrix and Classification Report:")
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        true_labels = [r['PredictedLabel'] for r in results]
        pred_labels = [generate_output(r['PredictedLabel'])[0] for r in results]
        cm = confusion_matrix(true_labels, pred_labels)
        cr = classification_report(true_labels, pred_labels)
        logger.info(f"Confusion Matrix:
{cm}")
        logger.info(f"Classification Report:
{cr}")
    except Exception as e:
        logger.error(f"Error generating confusion matrix or classification report: {e}")
    logger.info(output_df.head())
except Exception as e:
    logger.error(f"Error saving predictions to CSV: {e}")
