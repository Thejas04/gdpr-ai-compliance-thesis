import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from loguru import logger

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_FILE = os.path.join(PROJECT_ROOT, 'output', 'reports', 'evaluation_report.csv')
PLOT_PATH = os.path.join(PROJECT_ROOT, 'output', 'plots', 'performance_comparison.png')

sns.set(style="whitegrid")

# Ensure output and plot directories exist
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

# Check if the evaluation report file exists
if not os.path.exists(REPORT_FILE):
    logger.error(f"Evaluation report not found at {REPORT_FILE}")
    exit(1)

# Load evaluation report
df = pd.read_csv(REPORT_FILE)

# Plot performance metrics
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x="Model", y="Accuracy", marker="o", label="Accuracy")
sns.lineplot(data=df, x="Model", y="Precision", marker="o", label="Precision")
sns.lineplot(data=df, x="Model", y="Recall", marker="o", label="Recall")
sns.lineplot(data=df, x="Model", y="F1-Score", marker="o", label="F1-Score")
plt.title("Model Performance Metrics")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()
