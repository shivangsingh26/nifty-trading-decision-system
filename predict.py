"""
Production Prediction Script
Load trained model and make predictions on new data
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer


# Top 20 features (from production training)
TOP_20_FEATURES = [
    'macd', 'momentum_strength', 'vol_expansion', 'atr',
    'volatility', 'price_position', 'prev_return', 'bb_width',
    'intraday_return', 'macd_signal', 'return_3min', 'atr_pct',
    'dist_from_ema_50', 'return_5min', 'hl_range', 'rsi',
    'stoch_d', 'macd_diff', 'sma_10_20_cross', 'sma_5_10_cross'
]


def load_model(model_path='models/production_model.pkl'):
    """Load trained model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("   Please train the model first using: python train_production.py")
        return None


def prepare_features(data_path):
    """Prepare features from raw data"""
    print(f"\nLoading data from {data_path}...")

    # Load data
    data_loader = DataLoader(data_path)
    data_loader.load_data()
    data_loader.preprocess()
    df = data_loader.df

    # Create features
    print("Creating features...")
    feature_engineer = FeatureEngineer(
        df=df,
        sma_windows=config.SMA_WINDOWS,
        rsi_period=config.RSI_PERIOD,
        volatility_window=config.VOLATILITY_WINDOW,
        lag_periods=config.LAG_PERIODS
    )

    feature_engineer.create_price_features()
    feature_engineer.create_moving_averages()
    feature_engineer.create_rsi()
    feature_engineer.create_macd()
    feature_engineer.create_volatility_features()
    feature_engineer.create_lag_features()
    feature_engineer.create_time_features()
    feature_engineer.create_bollinger_bands()
    feature_engineer.create_atr()
    feature_engineer.create_stochastic()
    feature_engineer.create_roc()
    feature_engineer.create_advanced_features()

    df = feature_engineer.df.dropna().reset_index(drop=True)

    # Select top 20 features
    X = df[TOP_20_FEATURES]

    print(f"✅ Features prepared: {len(X)} samples, {len(TOP_20_FEATURES)} features")

    return X, df


def make_predictions(model, X, df, confidence_threshold=0.60):
    """Make predictions with confidence filtering"""
    print(f"\nMaking predictions (confidence threshold: {confidence_threshold*100:.0f}%)...")

    # Get probability predictions
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    # Calculate confidence (max probability)
    confidence = np.max(probabilities, axis=1)

    # Create results dataframe
    results = pd.DataFrame({
        'timestamp': df['timestamp'],
        'close': df['close'],
        'prediction': predictions,
        'down_prob': probabilities[:, 0],
        'up_prob': probabilities[:, 1],
        'confidence': confidence
    })

    # Map predictions to signals
    results['signal'] = results['prediction'].map({0: 'SELL', 1: 'BUY'})

    # Filter by confidence
    results['trade'] = np.where(results['confidence'] >= confidence_threshold,
                                results['signal'], 'WAIT')

    # Summary statistics
    total = len(results)
    buy_signals = (results['trade'] == 'BUY').sum()
    sell_signals = (results['trade'] == 'SELL').sum()
    wait_signals = (results['trade'] == 'WAIT').sum()

    avg_confidence = results[results['trade'] != 'WAIT']['confidence'].mean()

    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print('='*70)
    print(f"Total candles: {total}")
    print(f"BUY signals:   {buy_signals:6d} ({buy_signals/total*100:5.1f}%)")
    print(f"SELL signals:  {sell_signals:6d} ({sell_signals/total*100:5.1f}%)")
    print(f"WAIT (low confidence): {wait_signals:6d} ({wait_signals/total*100:5.1f}%)")
    print(f"\nAverage confidence: {avg_confidence:.2%}")
    print('='*70)

    return results


def display_recent_signals(results, n=10):
    """Display most recent trading signals"""
    print(f"\n{'='*70}")
    print(f"LAST {n} SIGNALS (Most Recent First)")
    print('='*70)

    recent = results.tail(n).iloc[::-1]  # Reverse to show most recent first

    for _, row in recent.iterrows():
        timestamp = row['timestamp']
        close = row['close']
        signal = row['trade']
        confidence = row['confidence']

        # Color coding
        if signal == 'BUY':
            indicator = '🟢'
        elif signal == 'SELL':
            indicator = '🔴'
        else:
            indicator = '⚪'

        print(f"{indicator} {timestamp} | {close:8.2f} | {signal:4s} | Conf: {confidence:.2%}")

    print('='*70)


def save_predictions(results, output_path='output/predictions.csv'):
    """Save predictions to CSV"""
    results.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to {output_path}")


def main():
    """Main prediction workflow"""
    parser = argparse.ArgumentParser(description='NIFTY Trading System - Predictions')
    parser.add_argument('--data', type=str, default=config.DATA_PATH,
                       help='Path to NIFTY data CSV')
    parser.add_argument('--model', type=str, default='models/production_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.60,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--output', type=str, default='output/predictions.csv',
                       help='Output path for predictions')
    parser.add_argument('--show', type=int, default=20,
                       help='Number of recent signals to display')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("NIFTY TRADING SYSTEM - PRODUCTION PREDICTIONS")
    print("="*70)

    # Load model
    model = load_model(args.model)
    if model is None:
        return

    # Prepare features
    X, df = prepare_features(args.data)

    # Make predictions
    results = make_predictions(model, X, df, args.confidence)

    # Display recent signals
    display_recent_signals(results, n=args.show)

    # Save predictions
    save_predictions(results, args.output)

    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Review predictions in {args.output}")
    print(f"2. Filter for BUY/SELL signals with high confidence")
    print(f"3. Apply risk management rules before trading")
    print(f"4. Paper trade before live trading!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
