# NIFTY Trading Decision System

A production-ready machine learning service to predict whether the next candle's closing price will go up or down using NIFTY intraday historical data.

## Project Overview

This project implements multiple ML models to predict price movements of NIFTY index based on 1-minute candle data. The solution includes:
- Binary classification (price up vs down)
- Multiple model comparison (Logistic Regression, Random Forest, LightGBM)
- Technical indicator feature engineering
- Trading signal generation with PnL calculation

---

## Project Structure

```
nifty-trading-decision-system/
│
├── main.py                    # Main execution script
├── config.py                  # Configuration parameters
│
├── src/                       # Source modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── feature_engineer.py    # Feature engineering
│   ├── model_trainer.py       # Model training and comparison
│   └── evaluator.py           # Evaluation and signal generation
│
├── models/                    # Saved trained models
│   └── best_model.pkl
│
├── output/                    # Generated outputs
│   ├── final_predictions.csv  # Final predictions with PnL
│   └── metrics_report.txt     # Detailed evaluation metrics
│
├── nifty_data.csv            # Input dataset (~319k rows)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── APPROACH.md              # Detailed methodology document
```

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone/Download the repository**
   ```bash
   cd nifty-trading-decision-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data file**
   - Ensure `nifty_data.csv` is present in the root directory
   - File should contain OHLC data with columns: `timestamp`, `open`, `high`, `low`, `close`

---

## Usage

### Run the complete pipeline (recommended)
```bash
python main.py
```
This will:
- Load and preprocess data
- Create features
- Train all 3 models
- Compare and select the best model
- Generate predictions and PnL
- Save outputs to `output/` directory

### Run with specific model
```bash
# Train only Logistic Regression
python main.py --model logistic_regression

# Train only Random Forest
python main.py --model random_forest

# Train only LightGBM
python main.py --model lightgbm
```

---

## Outputs

After execution, the following files will be generated:

1. **`output/final_predictions.csv`**
   - Contains test set with predictions and PnL
   - Columns: `Timestamp`, `Close`, `Predicted`, `model_call`, `model_pnl`

2. **`output/metrics_report.txt`**
   - Detailed evaluation metrics
   - Model comparison table
   - Confusion matrix
   - Classification report
   - PnL summary

3. **`models/best_model.pkl`**
   - Saved best performing model (can be loaded for future predictions)

---

## Methodology Overview

### 1. Data Preprocessing
- Sort by timestamp (chronological order)
- Create binary target: `1` if next close > current close, else `0`
- Handle missing values and duplicates

### 2. Feature Engineering (20+ features)
- **Price Features**: Intraday return, previous return, high-low range
- **Moving Averages**: SMA(5, 10, 20) and distance from SMA
- **Momentum Indicators**: RSI (14-period), MACD
- **Volatility**: Rolling standard deviation
- **Lag Features**: Previous 3 candles' close prices
- **Time Features**: Hour, minute (captures intraday patterns)

### 3. Train/Test Split
- **Time-based split** (70% train, 30% test)
- Prevents data leakage and respects temporal nature

### 4. Models Trained
- **Logistic Regression**: Simple linear baseline
- **Random Forest**: Ensemble of decision trees
- **LightGBM**: Gradient boosting (typically best performance)

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

### 6. Signal Generation & PnL
- `model_call = "buy"` if prediction = 1 (price up)
- `model_call = "sell"` if prediction = 0 (price down)
- Cumulative PnL calculated row-by-row

---

## Model Performance

**Best Model: LightGBM**

The LightGBM model performed best with the following characteristics:
- **Superior accuracy** compared to baseline models due to gradient boosting's ability to learn from errors iteratively
- **Handles complex patterns** in financial time series data effectively through tree-based splits
- **Feature importance insights** help identify which technical indicators contribute most to predictions
- **Computational efficiency** makes it suitable for large intraday datasets

---

## Configuration

All parameters can be adjusted in `config.py`:
- Train/test split ratio
- Feature engineering parameters (SMA windows, RSI period, etc.)
- Model hyperparameters
- File paths

---

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: ML models and metrics
- **lightgbm**: Gradient boosting
- **matplotlib**: Visualization (optional)

---

## Future Enhancements

- Add more advanced features (Bollinger Bands, ATR, etc.)
- Implement hyperparameter tuning (GridSearchCV)
- Add ensemble methods (stacking, voting)
- Implement walk-forward validation
- Add visualization dashboards

---

## Author

**Shivang Singh**

---

## Notes

- Stock price prediction is inherently difficult due to market randomness
- Accuracy above 55% is considered good for this type of problem
- Always validate strategies with out-of-sample data before live trading
- Past performance does not guarantee future results
