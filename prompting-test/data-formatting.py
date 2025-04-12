import pandas as pd
import re
import os
import shutil


OUTPUT_FOLDER = "output"
POST_OUTPUT_FOLDER = "post-output"
os.makedirs(POST_OUTPUT_FOLDER, exist_ok=True)


def extract_predicted_label(response):
    response = response.lower()
    if "compliant" in response:
        return "compliant"
    elif "non-compliant" in response or "not compliant" in response:
        return "non-compliant"
    elif "partially compliant" in response:
        return "partially compliant"
    else:
        return "unknown"


def extract_ground_truth(filename):
    filename = filename.lower()
    if "compliant" in filename:
        return "compliant"
    elif "non_compliant" in filename:
        return "non-compliant"
    elif "partially" in filename:
        return "partially compliant"
    else:
        return "unknown"


def format_prompting_output(input_path, output_path):
    data = pd.read_csv(input_path)
    data['PredictedLabel'] = data['Response'].apply(extract_predicted_label)
    data['GroundTruth'] = data['Filename'].apply(extract_ground_truth)
    formatted_data = data[['Filename', 'Model', 'Prompt', 'GroundTruth', 'PredictedLabel']]
    formatted_data.to_csv(output_path, index=False)
    print(f"Formatted dataset saved to {output_path}")


def process_all_files():
    for file_name in os.listdir(OUTPUT_FOLDER):
        input_path = os.path.join(OUTPUT_FOLDER, file_name)
        output_path = os.path.join(POST_OUTPUT_FOLDER, file_name)
        try:
            shutil.copy(input_path, POST_OUTPUT_FOLDER)
            format_prompting_output(input_path, output_path)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    process_all_files()
