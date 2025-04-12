# Script: main.py
from .fetch_outputs import fetch_output_files
from .data_loader import load_data
from .evaluation import main as evaluate
from .visualization import visualize
from loguru import logger

def main():
    logger.info("🚀 Starting the complete pipeline...")

    # Step 1: Fetch outputs before evaluating
    logger.info("🔄 Fetching outputs before evaluation...")
    try:
        fetch_output_files()
        logger.info("✅ Fetching completed.")
    except Exception as e:
        logger.error(f"❌ Error during fetching outputs: {str(e)}")

    # Step 2: Load and preprocess data
    logger.info("🔄 Loading and preprocessing data...")
    try:
        load_data()
        logger.info("✅ Data loading and preprocessing completed.")
    except Exception as e:
        logger.error(f"❌ Error during data loading: {str(e)}")
    
    # Step 3: Run evaluation
    logger.info("📝 Running evaluation...")
    try:
        evaluate()
        logger.info("✅ Evaluation completed.")
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {str(e)}")
    
    # Step 4: Run visualization
    logger.info("📊 Generating visualizations...")
    try:
        visualize()
        logger.info("✅ Visualization completed.")
    except Exception as e:
        logger.error(f"❌ Error during visualization: {str(e)}")

if __name__ == "__main__":
    main()
