import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Label map
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

# Inverse label map for better visualization
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Load data
def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

# Analyze class distribution
def analyze_class_distribution(data):
    labels = [item['label'] for item in data if item['label'] in LABEL_MAP]
    label_counts = Counter(labels)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['Count'])
    df = df.reset_index().rename(columns={'index': 'Class'})

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Count', y='Class', data=df, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Number of Samples')
    plt.ylabel('Class')
    plt.show()

# Analyze text lengths
def analyze_text_lengths(data):
    text_lengths = [len(item['text'].split()) for item in data]

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.histplot(text_lengths, bins=50, kde=True)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length (words)')
    plt.ylabel('Frequency')
    plt.show()

# Inspect sample data
def inspect_samples(data, num_samples=3):
    logger.info("Inspecting sample texts from each class...")
    samples = {label: [] for label in LABEL_MAP.keys()}
    for item in data:
        label = item['label']
        if len(samples[label]) < num_samples:
            samples[label].append(item['text'])

    for label, texts in samples.items():
        logger.info(f"\nClass: {label}")
        for text in texts:
            logger.info(f"Sample: {text[:300]}...")

# Main function to run analysis
def main():
    train_data = load_data('../datasets/training_data.json')
    val_data = load_data('../datasets/validation_data.json')

    logger.info("\nAnalyzing Training Data:")
    analyze_class_distribution(train_data)
    analyze_text_lengths(train_data)
    inspect_samples(train_data)

    logger.info("\nAnalyzing Validation Data:")
    analyze_class_distribution(val_data)
    analyze_text_lengths(val_data)
    inspect_samples(val_data)

if __name__ == "__main__":
    main()
