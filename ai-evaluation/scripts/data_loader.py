# Script: data_loader.py
import os
import pandas as pd
from loguru import logger

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_FOLDER = os.path.join(PROJECT_ROOT, 'input')
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'data_loader.log')

# Logging setup
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", level="INFO")

logger.info("📥 Starting data loading and preprocessing...")

# Load data files and perform basic checks
def load_data():
    try:
        csv_files = [file for file in os.listdir(INPUT_FOLDER) if file.endswith('.csv')]
        
        if not csv_files:
            logger.warning("⚠️ No CSV files found in the input folder.")
            return

        for file_name in csv_files:
            file_path = os.path.join(INPUT_FOLDER, file_name)
            logger.info(f"📂 Loading file: {file_path}")

            try:
                df = pd.read_csv(file_path)
                logger.info(f"✅ Loaded {file_name} with shape {df.shape}")

                # Basic data validation and cleanup
                if df.isnull().values.any():
                    logger.warning(f"⚠️ Missing values found in {file_name}. Filling with empty strings.")
                    df.fillna('', inplace=True)

                # Ensure consistent column names
                if 'category' in df.columns:
                    df.rename(columns={'category': 'Category'}, inplace=True)
                if 'prediction' in df.columns:
                    df.rename(columns={'prediction': 'Prediction'}, inplace=True)

                # Save the cleaned data back to the same file
                df.to_csv(file_path, index=False)
                logger.info(f"✅ Data cleaned and saved: {file_path}")

            except Exception as e:
                logger.error(f"❌ Failed to process {file_name}: {str(e)}")

        logger.info("✅ Data loading and preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"❌ Error during data loading: {str(e)}")

if __name__ == "__main__":
    load_data()
