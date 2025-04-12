# Robust Evaluation Script
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from loguru import logger
import time

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_FOLDER = os.path.join(PROJECT_ROOT, 'input')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output', 'reports')
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'evaluation.log')
REPORT_FILE = os.path.join(OUTPUT_FOLDER, 'evaluation_report.csv')

# Logging setup
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger.add(LOG_FILE, rotation='10 MB', level='INFO')

# Utility: Log message
logger.info('🚀 Starting robust evaluation...')

# Utility: Load CSV files
def load_csv_files():
    csv_files = []
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith('.csv'):
            csv_files.append(os.path.join(INPUT_FOLDER, file_name))
    if not csv_files:
        logger.error('No CSV files found in the input folder.')
    return csv_files

# Format Detection
def detect_format(df):
    if 'Category' in df.columns and 'Prediction' in df.columns:
        return 'standard'
    elif 'Label' in df.columns and 'Generated' in df.columns:
        return 'fine-tuning'
    elif 'Prompt' in df.columns and 'Category' in df.columns and 'Response' in df.columns:
        return 'prompt-based'
    else:
        logger.warning(f'⚠️ Unrecognized format with columns: {list(df.columns)}')
        return 'unknown'

# Transform to Common Format
def transform_to_common_format(df, format_type):
    try:
        if format_type == 'standard':
            df['Category'] = df['Category'].str.lower()
            df['Prediction'] = df['Prediction'].str.lower()
        elif format_type == 'fine-tuning':
            df['Category'] = df['Label'].str.lower()
            df['Prediction'] = df['Generated'].str.lower()
        elif format_type == 'prompt-based':
            df['Category'] = df['Category'].str.lower()
            df['Prediction'] = df['Response'].str.lower()
        else:
            logger.error(f'❌ Unsupported format type: {format_type}')
            return None
        logger.info(f'Transformed Data for format {format_type} from file: {df.head()}')
        return df[['Category', 'Prediction']]
    except Exception as e:
        logger.error(f'❌ Error transforming data format: {str(e)}')
        return None

# Evaluate Model Output
def evaluate_results(df):
    ground_truth = df['Category']
    predictions = df['Prediction']
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    class_report = classification_report(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)
    return accuracy, precision, recall, f1, class_report, conf_matrix

# Generate Report
def generate_report(results):
    try:
        report_df = pd.DataFrame(results)
        if report_df.empty:
            logger.error('❌ No valid results to save in the report.')
            return
        logger.info(f'📝 Saving the report to: {REPORT_FILE}')
        report_df.to_csv(REPORT_FILE, index=False)
        logger.info(f'✅ Report saved at: {REPORT_FILE}')
    except Exception as e:
        logger.error(f'❌ Error saving the report: {str(e)}')

# Main Function
def main():
    start_time = time.time()
    results = []
    csv_files = load_csv_files()
    for csv_file in tqdm(csv_files, desc='Evaluating Models'):
        try:
            logger.info(f'📝 Processing file: {csv_file}')
            df = pd.read_csv(csv_file)
            if df.empty:
                logger.error(f'❌ The file "{csv_file}" is empty. Skipping...')
                continue
            format_type = detect_format(df)
            if format_type == 'unknown':
                continue
            transformed_df = transform_to_common_format(df, format_type)
            if transformed_df is None:
                continue
            accuracy, precision, recall, f1, class_report, conf_matrix = evaluate_results(transformed_df)
            model_name = os.path.basename(csv_file).replace('.csv', '')
            results.append({'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})
            logger.info(f'✅ Model: {model_name}\n{class_report}')
            logger.info(f'✅ Confusion Matrix for {model_name}:\n{conf_matrix}')
        except Exception as e:
            logger.error(f'❌ Error processing {csv_file}: {str(e)}')
    generate_report(results)
    end_time = time.time()
    logger.info(f'✅ All evaluations completed in {end_time - start_time:.2f} seconds.')

if __name__ == '__main__':
    main()
