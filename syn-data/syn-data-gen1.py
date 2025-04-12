import ollama
import json
import random
import logging
import multiprocessing
from tqdm import tqdm  # Progress bar for better tracking
from concurrent.futures import ProcessPoolExecutor

MODEL_NAME = "deepseek-r1"
TOTAL_ENTRIES = 4000  # Target dataset size
LOG_FILE = "dataset_generation.log"
OUTPUT_FILE = "synthetic_gdpr_dataset.json"

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting GDPR dataset generation")

# Diverse Cookie Policy Attributes for Unique Scenario Generation
cookie_usage_purposes = ["advertising", "analytics", "personalization", "AI-powered profiling", "security tracking"]
consent_methods = ["explicit opt-in", "pre-checked box", "implied consent", "forced tracking", "no consent option"]
withdrawal_methods = ["clear opt-out button", "hidden opt-out", "email request required", "no withdrawal option"]
granular_controls = ["select categories", "accept/reject all only", "limited controls", "no granular choice"]
third_party_disclosures = ["fully disclosed", "partially mentioned", "not mentioned", "unclear language"]
risk_scores = ["Low", "Medium", "High"]

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

PROMPT_TEMPLATE = """
### GDPR Compliance Dataset Entry

Generate a **realistic and unique GDPR-compliant cookie policy text** based on the following scenario.
The policy should be written in **natural language**, typical of real-world GDPR cookie policy statements.
Avoid legal analysis, internal reflections, or reasoning.

Ensure the policy text meets the following criteria:
1. Uses natural, flowing language typical of GDPR cookie policies.
2. Avoids keyword stuffing and unnatural phrasing.
3. Includes real-world phrases and structured sentences.
4. Varies in length (short, medium, long) to simulate real scenarios.
5. Uses legal terminology appropriately and realistically.
6. Excludes any internal processing or reflection statements (e.g., <think> tags).
7. Varies in sentence complexity and vocabulary to enhance training diversity.

---
**Policy Text**: "{policy_text}"

Now, generate a **realistic cookie policy text** as per the GDPR requirements. The output should be in JSON format:

{{
  "text": "{policy_text}",
  "label": "{label}"
}}
---
"""


# Generate a realistic GDPR entry
def generate_gdpr_entry(label):
    try:
        length_type = random.choice(['short', 'medium', 'long'])
        num_sentences = {'short': random.randint(3, 5), 'medium': random.randint(8, 12), 'long': random.randint(15, 30)}[length_type]
        policy_text = " ".join(random.choices(cookie_usage_purposes + consent_methods + withdrawal_methods + granular_controls + third_party_disclosures, k=num_sentences))
        formatted_prompt = PROMPT_TEMPLATE.format(policy_text=policy_text, label=label)

        response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": formatted_prompt}])
        content = response.get("message", {}).get("content", "Error")

        # Filter out unwanted tags or reflections
        content = content.replace('<think>', '').replace('</think>', '').strip()

        return {"text": content, "label": label}
    except Exception as e:
        logging.error(f"Error generating GDPR entry for label {label}: {str(e)}")
        return {"text": "Generation failed.", "label": "Error"}


# Generate preview entries for each label
def generate_preview():
    print("\nGenerating Preview Entries...\n")
    for label in LABEL_MAP.keys():
        print(f"Preview for '{label}':")
        entry = generate_gdpr_entry(label)
        print(json.dumps(entry, indent=4))


# Save dataset to file
def save_dataset():
    print("Generating GDPR Compliance Dataset...")
    dataset = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("[\n")
        with tqdm(total=TOTAL_ENTRIES, desc="Generating Entries", unit="entry") as pbar:
            for label in LABEL_MAP.keys():
                for _ in range(500):
                    entry = generate_gdpr_entry(label)
                    dataset.append(entry)
                    json.dump(entry, f, indent=4)
                    f.write(",\n")
                    pbar.update(1)
        f.write("\n]")
    print(f"\nGDPR Dataset with {len(dataset)} entries saved to `{OUTPUT_FILE}`!")


# Generate preview and get user approval
generate_preview()
user_input = input("\nApprove and continue? (yes/no): ").strip().lower()
if user_input != "yes":
    print("\nGeneration aborted by user.")
    exit()

save_dataset()
