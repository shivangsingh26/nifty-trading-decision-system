"""
Configuration file for NIFTY Price Prediction Project
Contains all parameters and paths in one place for easy tuning
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "nifty_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Output files
FINAL_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "final_predictions.csv")
METRICS_REPORT_PATH = os.path.join(OUTPUT_DIR, "metrics_report.txt")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance.png")

# Data processing parameters
TRAIN_TEST_SPLIT_RATIO = 0.7  # 70% train, 30% test
RANDOM_STATE = 42  # For reproducibility

# Feature engineering parameters
SMA_WINDOWS = [5, 10, 20]  # Simple Moving Average windows
RSI_PERIOD = 14  # RSI period
VOLATILITY_WINDOW = 5  # Rolling volatility window
LAG_PERIODS = 3  # Number of lag features

# Model parameters
MODELS_TO_TRAIN = ['logistic_regression', 'random_forest', 'lightgbm']

# Logistic Regression parameters
LR_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'solver': 'lbfgs'
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# LightGBM parameters
LGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

# Evaluation parameters
METRICS_TO_CALCULATE = ['accuracy', 'precision', 'recall', 'f1']

# Display settings
PRINT_DETAILED_METRICS = True
SAVE_FEATURE_IMPORTANCE = True