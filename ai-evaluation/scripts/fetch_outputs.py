# Script: fetch_outputs.py
import os
import json
import shutil
from datetime import datetime
from loguru import logger

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'fetch_config.json')

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, config["output_folder"])
LOG_FILE = os.path.join(PROJECT_ROOT, config["log_file"])

# Logging setup
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", level="INFO")

def fetch_output_files():
    """
    Fetches output files from the source folders as specified in the configuration.
    Copies files only if they do not already exist in the target folder.
    """
    logger.info("🔄 Starting output file fetching...")

    for source in config["source_folders"]:
        folder_name = source.get("name")
        folder_path = os.path.expanduser(source.get("path"))

        if not os.path.exists(folder_path):
            logger.error(f"❌ Source folder '{folder_path}' not found. Skipping...")
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                src_file = os.path.join(folder_path, filename)
                dest_file = os.path.join(OUTPUT_FOLDER, filename)

                # Check if the file already exists to avoid duplication
                if os.path.exists(dest_file):
                    logger.info(f"✅ File '{filename}' already exists. Skipping fetch.")
                    continue

                try:
                    shutil.copy2(src_file, dest_file)
                    logger.info(f"✅ Successfully copied '{filename}' from '{folder_name}'")
                except Exception as e:
                    logger.error(f"❌ Error copying '{filename}': {str(e)}")

    logger.info("✅ Output file fetching completed.")
