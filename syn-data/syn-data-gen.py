import ollama
import json
import random
from tqdm import tqdm  # Progress bar for better tracking

MODEL_NAME = "deepseek-r1"
TOTAL_ENTRIES = 6000  # Target dataset size

# **Diverse Cookie Policy Attributes for Unique Scenario Generation**
cookie_usage_purposes = ["advertising", "analytics", "personalization", "AI-powered profiling", "security tracking"]
consent_methods = ["explicit opt-in", "pre-checked box", "implied consent", "forced tracking", "no consent option"]
withdrawal_methods = ["clear opt-out button", "hidden opt-out", "email request required", "no withdrawal option"]
granular_controls = ["select categories", "accept/reject all only", "limited controls", "no granular choice"]
third_party_disclosures = ["fully disclosed", "partially mentioned", "not mentioned", "unclear language"]
risk_scores = ["Low", "Medium", "High"]

PROMPT_TEMPLATE = """
### GDPR Compliance Dataset Entry

Generate a **unique** GDPR compliance dataset entry by classifying the following website cookie policy based on **GDPR Article 4(11) (Consent Definition) and Article 7 (Conditions for Consent).**

---
**Policy Text**: "{policy_text}"

Now, generate a **new GDPR compliance dataset entry** in the following format:

---
**Classification**: [Fully GDPR Compliant / Fully GDPR Non-Compliant / Partially Compliant]

**Structured Explanation**:
- **Explicit Consent Given?** [Yes/No + Justification]
- **Withdrawal Option Available?** [Yes/No + Justification]
- **Granular Consent Provided?** [Yes/No + Justification]
- **Third-Party Data Usage Transparency?** [Yes/No + Justification]
- **Legal Compliance (Article 4(11) & 7)**: [Compliant/Non-Compliant]
- **Final Justification**: [Concise explanation of the compliance level]

**Categories**: [Article 4(11), Article 7]
**Region**: [EU]
**Risk Score**: [High, Medium, Low]
**Remediation Suggestions**: [How to fix non-compliance]

---
Ensure that each generated entry is **unique**, avoids repetition, and represents a **realistic** GDPR compliance scenario.
"""

# **Function to Generate a GDPR Entry**
def generate_gdpr_entry():
    policy_text = (
        f"This website collects cookies for {random.choice(cookie_usage_purposes)}. "
        f"Users must {random.choice(['accept all cookies', 'opt-in explicitly', 'continue browsing to consent', 'have no choice to opt-out'])}. "
        f"Consent is handled via {random.choice(consent_methods)}."
    )
    formatted_prompt = PROMPT_TEMPLATE.format(policy_text=policy_text)
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": formatted_prompt}])

    if "message" in response and "content" in response["message"]:
        content = response["message"]["content"]
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()
        return content
    else:
        return "Error: Unexpected response format from Ollama"


print("\nContinuing with full dataset generation...\n")

output_file = "gdpr_compliance_dataset.json"
dataset = []

with open(output_file, "w", encoding="utf-8") as f:
    f.write("[\n")  # Start JSON array
    for i in tqdm(range(TOTAL_ENTRIES), desc="Generating GDPR Dataset", unit="entry"):
        entry = generate_gdpr_entry()
        dataset.append(entry)
        json.dump(entry, f, indent=4)
        if i < TOTAL_ENTRIES - 1:
            f.write(",\n")  # Add comma for JSON array formatting
    f.write("\n]")  # Close JSON array

print(f"\nGDPR Dataset with {len(dataset)} entries saved to `{output_file}`!")
