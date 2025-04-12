import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from docx import Document

# Corrected output directory path
model_dir = "./output/"

# Check if the model directory exists and contains necessary files
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

# Load the model and tokenizer from the fine-tuned model directory
print(" Loading fine-tuned model...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=9)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(" Model and tokenizer loaded successfully!")
except Exception as e:
    print(f" Failed to load model or tokenizer: {str(e)}")
    exit(1)

# Helper function to chunk long texts
def chunk_text(text, max_length=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

# Helper function for structured explanations based on detailed criteria
def get_structured_explanation(prediction, text):
    if prediction == 0:  # Fully Non-Compliant
        return {
            "Classification": "Fully GDPR Non-Compliant",
            "Structured Explanation": (
                "• Explicit Consent Given? No (The policy assumes consent rather than explicitly requesting it)."
                "\n• Withdrawal Option Available? No (No easy way to change settings after initial consent is given)."
                "\n• Granular Consent Provided? No (Users must accept/reject all cookies, with no granular control)."
                "\n• Third-Party Data Usage Transparency? No (Mentions third-party data collection but does not allow opt-out)."
                "\n• Legal Compliance (GDPR Articles 4(11) & 7)? No (Fails to ensure explicit, informed, and freely given consent)."
                "\nFinal Justification: The policy violates GDPR requirements by not explicitly requesting consent, not offering an easy withdrawal mechanism, and lacking granular control over cookie preferences."
            )
        }
    elif prediction == 1:  # Fully Compliant
        return {
            "Classification": "Fully GDPR Compliant",
            "Structured Explanation": (
                "• Explicit Consent Given? Yes (Users can choose between Accept All, Reject Everything, and Adjust Options)."
                "\n• Withdrawal Option Available? Yes (Users can change settings at any time via “Cookie and Advertising Settings”)."
                "\n• Granular Consent Provided? Yes (Users can select specific cookie categories instead of just Accept/Reject)."
                "\n• Third-Party Data Usage Transparency? Yes (Clearly mentions third-party cookies, data collection, and opt-out options)."
                "\n• Legal Compliance (GDPR Articles 4(11) & 7)? Yes (Meets GDPR requirements for explicit, informed, and freely given consent)."
                "\nFinal Justification: This policy ensures full compliance by providing clear, specific, and easily adjustable consent mechanisms, making it GDPR-compliant."
            )
        }
    elif prediction == 2:  # Partially Compliant (Article 4(11) Compliant, Article 7 Not)
        return {
            "Classification": "Partially GDPR Compliant (Article 4(11) Compliant, Article 7 Not)",
            "Structured Explanation": (
                "• Explicit Consent Given? Yes (Users can opt-in to cookies explicitly)."
                "\n• Withdrawal Option Available? No (Users cannot withdraw consent in a simple and accessible way)."
                "\n• Granular Consent Provided? Yes (Users can select specific types of cookies)."
                "\n• Third-Party Data Usage Transparency? Yes (Provides detailed information on third-party cookies)."
                "\n• Legal Compliance:"
                "\n  • Article 4(11) (Consent Definition): Compliant (Explicit and informed consent is given)."
                "\n  • Article 7 (Conditions for Consent): Non-Compliant (Fails to provide an easy withdrawal mechanism)."
                "\nFinal Justification: While the policy collects consent correctly, it does not comply with GDPR Article 7 as users cannot withdraw consent easily. A clear withdrawal option must be provided."
            )
        }
    elif prediction == 3:  # Partially Compliant (Article 4(11) Non-Compliant, Article 7 Compliant)
        return {
            "Classification": "Partially GDPR Compliant (Article 4(11) Non-Compliant, Article 7 Compliant)",
            "Structured Explanation": (
                "• Explicit Consent Given? No (Consent is either implied or vague, such as pre-checked boxes)."
                "\n• Withdrawal Option Available? Yes (Users can withdraw consent easily)."
                "\n• Granular Consent Provided? Yes (Users can select specific cookie categories)."
                "\n• Third-Party Data Usage Transparency? Yes (Clearly mentions third-party data usage)."
                "\n• Legal Compliance:"
                "\n  • Article 4(11) (Consent Definition): Non-Compliant (Consent is not clearly specified)."
                "\n  • Article 7 (Conditions for Consent): Compliant (Clear and simple withdrawal option)."
                "\nFinal Justification: The policy allows for easy withdrawal, but it violates Article 4(11) as consent is not clearly given."
            )
        }
    elif prediction == 4:  # Partially Compliant (Article 4(11) & Article 7 Partial Violation)
        return {
            "Classification": "Partially GDPR Compliant (Article 4(11) & Article 7 Partial Violation)",
            "Structured Explanation": (
                "• Explicit Consent Given? Yes (Consent is given, but it lacks sufficient clarity and specificity)."
                "\n• Withdrawal Option Available? No (Withdrawal is possible but not easy to find or use)."
                "\n• Granular Consent Provided? Yes (Users can choose categories of cookies)."
                "\n• Third-Party Data Usage Transparency? Yes (Policy explains third-party data usage)."
                "\n• Legal Compliance:"
                "\n  • Article 4(11) (Consent Definition): Compliant (Explicit consent given but not sufficiently specific)."
                "\n  • Article 7 (Conditions for Consent): Non-Compliant (Withdrawal option difficult to use)."
                "\nFinal Justification: While users can provide consent, the policy does not make withdrawal easy, violating Article 7."
            )
        }
    elif prediction == 5:  # Partially Compliant (Granular Consent Missing, Withdrawal Available)
        return {
            "Classification": "Partially GDPR Compliant (Granular Consent Missing, Withdrawal Available)",
            "Structured Explanation": (
                "• Explicit Consent Given? Yes (Users explicitly accept cookies)."
                "\n• Withdrawal Option Available? Yes (Users can revoke consent easily)."
                "\n• Granular Consent Provided? No (Users cannot select specific cookies; must accept all)."
                "\n• Third-Party Data Usage Transparency? Yes (The policy clearly explains third-party data usage)."
                "\n• Legal Compliance:"
                "\n  • Article 4(11) (Consent Definition): Non-Compliant (Fails to allow granular cookie selection)."
                "\n  • Article 7 (Conditions for Consent): Compliant (Withdrawal option provided)."
                "\nFinal Justification: While the policy allows users to withdraw consent easily, it lacks granular cookie selection, which violates Article 4(11)."
            )
        }
    elif prediction == 6:  # Partially Compliant (No Clear Consent, But Withdrawal Available)
        return {
            "Classification": "Partially GDPR Compliant (No Clear Consent, But Withdrawal Available)",
            "Structured Explanation": (
                "• Explicit Consent Given? No (Consent is unclear or implied)."
                "\n• Withdrawal Option Available? Yes (Easy process to withdraw consent)."
                "\n• Granular Consent Provided? Yes (Users can select specific cookies)."
                "\n• Third-Party Data Usage Transparency? Yes (Clear explanation on third-party usage)."
                "\n• Legal Compliance:"
                "\n  • Article 4(11) (Consent Definition): Non-Compliant (Consent is implied and not clear)."
                "\n  • Article 7 (Conditions for Consent): Compliant (Easy withdrawal)."
                "\nFinal Justification: The policy violates Article 4(11) because the consent process is unclear, even though withdrawal is easy."
            )
        }
    elif prediction == 7:  # Partially Compliant (Consent Given but Consent Process Is Difficult)
        return {
            "Classification": "Partially GDPR Compliant (Consent Given but Consent Process Is Difficult)",
            "Structured Explanation": (
                "• Explicit Consent Given? Yes (Users provide consent, but the process is complicated)."
                "\n• Withdrawal Option Available? No (Difficult or unclear process to withdraw consent)."
                "\n• Granular Consent Provided? No (Granular selection not available)."
                "\n• Third-Party Data Usage Transparency? No (No clear information on third-party usage)."
                "\n• Legal Compliance:"
                "\n  • Article 4(11) (Consent Definition): Compliant (Explicit consent provided, though difficult process)."
                "\n  • Article 7 (Conditions for Consent): Non-Compliant (Withdrawal is hard)."
                "\nFinal Justification: While consent is given, the complicated process violates Article 7 by making withdrawal difficult."
            )
        }
    elif prediction == 8:  # Partially Compliant (Consent Issues and Transparency Issues)
        return {
            "Classification": "Partially GDPR Compliant (Consent Issues and Transparency Issues)",
            "Structured Explanation": (
                "• Explicit Consent Given? No (The process for obtaining consent is unclear)."
                "\n• Withdrawal Option Available? No (Withdrawal process is not easily accessible)."
                "\n• Granular Consent Provided? No (Granular consent is not available)."
                "\n• Third-Party Data Usage Transparency? No (Lack of transparency regarding third-party data usage)."
                "\n• Legal Compliance:"
                "\n  • Article 4(11) (Consent Definition): Non-Compliant (Consent is not clear)."
                "\n  • Article 7 (Conditions for Consent): Non-Compliant (Withdrawal is not clear)."
                "\nFinal Justification: The policy violates both Articles 4(11) and 7 due to unclear consent, lack of transparency, and difficult withdrawal."
            )
        }

# Testing configurations
test_dir = "../testing/datasets/"
output_file = "./output/test_results.csv"
results = []

# Label mapping for predictions (expanded to 9 scenarios)
label_map = {
    0: "Fully GDPR Non-Compliant",
    1: "Fully GDPR Compliant",
    2: "Partially Compliant (Article 4(11) Compliant, Article 7 Not)",
    3: "Partially Compliant (Article 4(11) Non-Compliant, Article 7 Compliant)",
    4: "Partially Compliant (Article 4(11) & Article 7 Partial Violation)",
    5: "Partially Compliant (Granular Consent Missing, Withdrawal Available)",
    6: "Partially Compliant (No Clear Consent, But Withdrawal Available)",
    7: "Partially Compliant (Consent Given but Consent Process Is Difficult)",
    8: "Partially Compliant (Consent Issues and Transparency Issues)"
}

print(" Starting model testing...")
for folder in ["compliant", "non_compliant", "partially_compliant"]:
    folder_path = os.path.join(test_dir, folder)
    if not os.path.exists(folder_path):
        print(f"⚠️Warning: Test folder '{folder_path}' not found, skipping.")
        continue

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".docx"):
            file_path = os.path.join(folder_path, file)
            try:
                # Use python-docx to read the .docx file content
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])

                # Handle long texts by splitting into chunks
                chunks = chunk_text(text)

                # Aggregate predictions from all chunks
                chunk_predictions = []

                for chunk in chunks:
                    # Tokenize and make predictions
                    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                    chunk_predictions.append(prediction)

                # Aggregate chunk predictions (majority vote)
                final_prediction = max(set(chunk_predictions), key=chunk_predictions.count)
                predicted_label = label_map.get(final_prediction, "Unknown")

                # Get structured explanation based on prediction
                explanation = get_structured_explanation(final_prediction, text)

                # Log the result in the required format
                results.append({
                    "Prompt": "fine_tune",
                    "Category": folder,
                    "Filename": file,
                    "Model": "fine_tune",
                    "Response": predicted_label,
                    "Structured Explanation": explanation['Structured Explanation']
                })
                print(f"✅ Processed file: {file} -> Prediction: {predicted_label}")

            except Exception as e:
                print(f" Error processing file '{file}': {str(e)}")

# Save results to CSV in the required format
try:
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f" Testing complete! Results saved to {output_file}")
except Exception as e:
    print(f" Error saving results to CSV: {str(e)}")
