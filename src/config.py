"""
Configuration settings for the Kenyan Fraud Detection project.
This module loads environment variables and provides configuration constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model configuration
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
CROSS_VALIDATION_FOLDS = int(os.getenv("CROSS_VALIDATION_FOLDS", 5))

# Neural Network parameters
NN_EPOCHS = int(os.getenv("NN_EPOCHS", 100))
NN_BATCH_SIZE = int(os.getenv("NN_BATCH_SIZE", 32))
NN_LEARNING_RATE = float(os.getenv("NN_LEARNING_RATE", 0.001))

# XGBoost parameters
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", 6))
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", 0.1))
XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", 100))

# Random Forest parameters
RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", 100))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", 10))
RF_MIN_SAMPLES_SPLIT = int(os.getenv("RF_MIN_SAMPLES_SPLIT", 2))

# Dataset configuration
DATASET_COLUMNS = {
    "categorical": [
        "location", 
        "transaction_type", 
        "merchant_category", 
        "age_group"
    ],
    "numerical": [
        "transaction_amount", 
        "transaction_frequency", 
        "time_since_last_txn_hrs", 
        "distance_from_home_km", 
        "card_age_months"
    ],
    "target": "is_fraud"
}

# Feature engineering configuration
FEATURE_ENGINEERING = {
    "scaling": ["transaction_amount", "transaction_frequency", "distance_from_home_km"],
    "encoding": ["location", "transaction_type", "merchant_category"],
    "binning": {"age": [18, 25, 35, 45, 55, 65, 100]}
}
