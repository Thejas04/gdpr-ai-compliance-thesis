import os
import json
import signal
import subprocess
import pandas as pd
from docx import Document
from tqdm import tqdm
import ollama
from datetime import datetime

# Load configuration
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

PROMPT_FOLDER = config["prompt_folder"]
DATA_FOLDER = config["data_folder"]
OUTPUT_FILE = config["output_file"]
LOG_FILE = config["log_file"]
BATCH_SIZE = config["batch_size"]

# Global variable to track progress and results
results = []
interrupted = False

# Logging setup
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
def log_message(message):
    """Log messages to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")
    print(message)

# Graceful exit on interruption
def handle_interrupt(sig, frame):
    global interrupted
    interrupted = True
    log_message("🔴 Process interrupted by user. Saving current progress...")
    save_results()

signal.signal(signal.SIGINT, handle_interrupt)

def get_all_prompts(prompt_folder):
    """Load all prompts from the gdpr_prompts folder."""
    prompts = {}
    try:
        for prompt_file in sorted(os.listdir(prompt_folder)):
            if prompt_file.endswith(".txt"):
                prompt_name = os.path.splitext(prompt_file)[0]
                prompt_path = os.path.join(prompt_folder, prompt_file)
                with open(prompt_path, "r", encoding="utf-8") as file:
                    prompts[prompt_name] = file.read()
        return prompts
    except Exception as e:
        log_message(f"❌ Error loading prompts from '{prompt_folder}': {e}")
        return {}

def load_cookie_policies(data_folder):
    """Load cookie policy documents from the data folder."""
    policies = []
    for folder_name in ["compliant", "non_compliant", "partially-compliant"]:
        folder_path = os.path.join(data_folder, folder_name)
        if not os.path.exists(folder_path):
            log_message(f"❌ Folder '{folder_path}' does not exist. Skipping...")
            continue
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".docx"):
                doc_path = os.path.join(folder_path, filename)
                try:
                    doc = Document(doc_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    policies.append((folder_name, filename, text))
                    log_message(f"✅ Loaded cookie policy from '{filename}' in '{folder_name}' folder.")
                except Exception as e:
                    log_message(f"❌ Error reading '{filename}': {e}")
    return policies

def list_models():
    """List all available models using the Ollama CLI."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            models = result.stdout.strip().split("\n")
            model_names = [model.split()[0] for model in models if model]
            return model_names
        else:
            log_message(f"❌ Error listing models: {result.stderr}")
            return []
    except Exception as e:
        log_message(f"❌ Exception during model listing: {e}")
        return []

def pull_model(model_name):
    """Pull a model using the Ollama CLI."""
    try:
        print(f"🚀 Pulling model '{model_name}'...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Successfully pulled model '{model_name}'.")
            log_message(f"✅ Model '{model_name}' pulled successfully.")
            return True
        else:
            print(f"❌ Error pulling model '{model_name}': {result.stderr}")
            log_message(f"❌ Error pulling model '{model_name}': {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception occurred while pulling model '{model_name}': {e}")
        log_message(f"❌ Exception occurred while pulling model '{model_name}': {e}")
        return False

def choose_model():
    """Prompt user to select a model or pull a new one."""
    available_models = list_models()
    if not available_models:
        print("❌ No models are currently available locally.")
    else:
        print("✅ Available Models:")
        for idx, model in enumerate(available_models, 1):
            print(f"{idx}. {model}")

    print(f"{len(available_models) + 1}. Pull a new model")

    choice = input("\nSelect a model number or choose to pull a new model: ").strip()

    if choice.isdigit():
        choice = int(choice)
        if 1 <= choice <= len(available_models):
            return available_models[choice - 1]
        elif choice == len(available_models) + 1:
            model_name = input("Enter the name of the model to pull: ").strip()
            if pull_model(model_name):
                print(f"✅ Model '{model_name}' pulled successfully.")
                return model_name
            else:
                print("❌ Failed to pull the specified model.")
                return None
        else:
            print("❌ Invalid choice. Please try again.")
            return choose_model()
    else:
        print("❌ Invalid input. Please enter a number.")
        return choose_model()

def run_model(model_name, prompt):
    """Run a model using the Ollama Python API."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        else:
            log_message(f"❌ Unexpected response format from Ollama: {response}")
            return "❌ Error: Unexpected response format from Ollama"
    except Exception as e:
        log_message(f"❌ Error running model '{model_name}': {e}")
        return "Error"

def save_results():
    """Save the results to a CSV file with model-specific naming."""
    try:
        model_specific_file = OUTPUT_FILE.replace(".csv", f"_{model_name}.csv")
        df = pd.DataFrame(results)
        output_path = model_specific_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        log_message(f"✅ Results saved to '{output_path}'")
    except Exception as e:
        log_message(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    model_name = choose_model()
    if not model_name:
        log_message("❌ Model selection failed. Exiting.")
        exit()

    prompt_folder = os.path.join(PROMPT_FOLDER, "gdpr_prompts")
    prompts = get_all_prompts(prompt_folder)
    if not prompts:
        log_message("❌ No prompts found. Exiting.")
        exit()

    policies = load_cookie_policies(DATA_FOLDER)
    if not policies:
        log_message("❌ No cookie policies found.")
        exit()

    log_message("🚀 Starting model runs...")
    with tqdm(total=len(prompts) * len(policies), desc="Processing Policies", unit="policy") as pbar:
        for prompt_name, prompt_template in prompts.items():
            for idx, (category, filename, policy_text) in enumerate(policies):
                if interrupted:
                    break
                prompt = prompt_template.format(cookie_policy=policy_text)
                response = run_model(model_name, prompt)
                results.append({
                    "Prompt": prompt_name,
                    "Category": category,
                    "Filename": filename,
                    "Model": model_name,
                    "Response": response
                })
                if (idx + 1) % BATCH_SIZE == 0:
                    save_results()
                pbar.update(1)

    save_results()
    log_message("✅ Processing complete.")
