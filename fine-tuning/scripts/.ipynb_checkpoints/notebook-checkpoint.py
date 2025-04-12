import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics
with open('../output/metrics.json', 'r') as f:
    metrics = json.load(f)

# Load confusion matrix
cm = pd.read_csv('../output/confusion_matrix.csv', index_col=0)

# Display classification report
print("Classification Report:")
for label, scores in metrics.items():
    print(f"{label}:")
    if isinstance(scores, dict):  # Check if the value is a dictionary
        for metric, value in scores.items():
            print(f"  {metric}: {value}")
        print("")
    else:
        print(f"  {scores}")

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Visualize metrics (Accuracy, Precision, Recall, F1)
metrics_df = pd.DataFrame(metrics).T
metrics_df = metrics_df[['precision', 'recall', 'f1-score', 'support']]

# Plot metrics
plt.figure(figsize=(10, 6))
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title('Model Performance Metrics')
plt.xlabel('Labels')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()

# Plot support distribution
plt.figure(figsize=(10, 6))
metrics_df['support'].plot(kind='bar', color='gray')
plt.title('Support Distribution Across Labels')
plt.xlabel('Labels')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.show()
