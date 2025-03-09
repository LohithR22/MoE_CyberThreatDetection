# filepath: /moe-cybersecurity-detection/moe-cybersecurity-detection/src/main.py
import os
import sys
import logging
from model import MoEModel
from train import train_model
from utils import load_data

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Load datasets from the specified folder
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'DatasetsForCyberThreat')
        train_data, val_data = load_data(dataset_path)
        logger.info("Datasets loaded successfully.")

        # Initialize the Mixture of Experts model
        model = MoEModel()
        logger.info("Model initialized successfully.")

        # Start the training process
        train_model(model, train_data, val_data)
        logger.info("Training process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()