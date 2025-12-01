"""
Feature Engineering Module
Creates technical indicators and features for model training
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Generates features from OHLC data for ML models
    """

    def __init__(self, df, sma_windows=[5, 10, 20], rsi_period=14,
                 volatility_window=5, lag_periods=3):
        """
        Initialize FeatureEngineer

        Args:
            df (pd.DataFrame): Input dataframe with OHLC data
            sma_windows (list): Windows for Simple Moving Averages
            rsi_period (int): Period for RSI calculation
            volatility_window (int): Window for volatility calculation
            lag_periods (int): Number of lag features to create
        """
        self.df = df.copy()
        self.sma_windows = sma_windows
        self.rsi_period = rsi_period
        self.volatility_window = volatility_window
        self.lag_periods = lag_periods
        self.feature_columns = []

    def create_price_features(self):
        """
        Create basic price-based features:
        - Intraday return: (close - open) / open
        - Previous candle return: (close - prev_close) / prev_close
        - High-Low range: (high - low) / close
        """
        print("Creating price-based features...")

        # Intraday return (candle return)
        self.df['intraday_return'] = (self.df['close'] - self.df['open']) / self.df['open']

        # Return from previous candle
        self.df['prev_close'] = self.df['close'].shift(1)
        self.df['prev_return'] = (self.df['close'] - self.df['prev_close']) / self.df['prev_close']

        # High-Low range (volatility proxy)
        self.df['hl_range'] = (self.df['high'] - self.df['low']) / self.df['close']

        self.feature_columns.extend(['intraday_return', 'prev_return', 'hl_range'])

    def create_moving_averages(self):
        """
        Create Simple Moving Averages (SMA) and distance from SMA
        """
        print(f"Creating moving averages: {self.sma_windows}...")

        for window in self.sma_windows:
            sma_col = f'sma_{window}'
            dist_col = f'dist_from_sma_{window}'

            # Calculate SMA
            self.df[sma_col] = self.df['close'].rolling(window=window).mean()

            # Distance from SMA (normalized)
            self.df[dist_col] = (self.df['close'] - self.df[sma_col]) / self.df[sma_col]

            self.feature_columns.extend([sma_col, dist_col])

    def create_rsi(self):
        """
        Create Relative Strength Index (RSI)
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        print(f"Creating RSI ({self.rsi_period}-period)...")

        # Calculate price changes
        delta = self.df['close'].diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate average gain and loss using exponential moving average
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

        self.feature_columns.append('rsi')

    def create_macd(self):
        """
        Create MACD (Moving Average Convergence Divergence)
        MACD = 12-period EMA - 26-period EMA
        Signal = 9-period EMA of MACD
        """
        print("Creating MACD...")

        # Calculate EMAs
        ema_12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.df['close'].ewm(span=26, adjust=False).mean()

        # MACD line
        self.df['macd'] = ema_12 - ema_26

        # Signal line
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()

        # MACD histogram
        self.df['macd_diff'] = self.df['macd'] - self.df['macd_signal']

        self.feature_columns.extend(['macd', 'macd_signal', 'macd_diff'])

    def create_volatility_features(self):
        """
        Create volatility-based features:
        - Rolling standard deviation of returns
        """
        print(f"Creating volatility features ({self.volatility_window}-period)...")

        # Calculate rolling volatility
        self.df['volatility'] = self.df['prev_return'].rolling(
            window=self.volatility_window
        ).std()

        self.feature_columns.append('volatility')

    def create_lag_features(self):
        """
        Create lag features (previous N candles' close prices)
        """
        print(f"Creating lag features (previous {self.lag_periods} candles)...")

        for i in range(1, self.lag_periods + 1):
            lag_col = f'close_lag_{i}'
            self.df[lag_col] = self.df['close'].shift(i)
            self.feature_columns.append(lag_col)

    def create_time_features(self):
        """
        Extract time-based features from timestamp:
        - Hour of day
        - Minute of hour
        - Hour-specific flags (opening, closing, lunch)
        """
        print("Creating time-based features...")

        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['minute'] = self.df['timestamp'].dt.minute

        # Hour-specific patterns (critical for intraday!)
        self.df['is_opening_hour'] = (self.df['hour'] == 9).astype(int)
        self.df['is_closing_hour'] = (self.df['hour'] == 15).astype(int)
        self.df['is_lunch_hour'] = ((self.df['hour'] >= 12) & (self.df['hour'] <= 13)).astype(int)

        self.feature_columns.extend(['hour', 'minute', 'is_opening_hour',
                                     'is_closing_hour', 'is_lunch_hour'])

    def create_advanced_features(self):
        """
        Create advanced features with higher predictive power:
        - Momentum strength (not just direction)
        - Multi-timeframe returns
        - Price position in range
        - Volatility expansion/contraction
        """
        print("Creating advanced features...")

        # A. Momentum strength (absolute value matters!)
        self.df['momentum_strength'] = self.df['prev_return'].abs()
        self.df['intraday_strength'] = self.df['intraday_return'].abs()

        # B. Price position in candle range (0 = at low, 1 = at high)
        range_diff = self.df['high'] - self.df['low']
        range_diff = range_diff.replace(0, 0.0001)  # Avoid division by zero
        self.df['price_position'] = (self.df['close'] - self.df['low']) / range_diff

        # C. Multi-timeframe returns (capture broader context)
        self.df['return_3min'] = self.df['close'].pct_change(3) * 100
        self.df['return_5min'] = self.df['close'].pct_change(5) * 100
        self.df['return_10min'] = self.df['close'].pct_change(10) * 100

        # D. Volatility expansion/contraction
        hl_range_prev = self.df['hl_range'].shift(1)
        hl_range_prev = hl_range_prev.replace(0, 0.0001)  # Avoid division by zero
        self.df['vol_expansion'] = self.df['hl_range'] / hl_range_prev

        # E. Moving average crossovers
        self.df['sma_5_10_cross'] = self.df['sma_5'] - self.df['sma_10']
        self.df['sma_10_20_cross'] = self.df['sma_10'] - self.df['sma_20']

        # F. RSI zones
        self.df['rsi_oversold'] = (self.df['rsi'] < 30).astype(int)
        self.df['rsi_overbought'] = (self.df['rsi'] > 70).astype(int)

        self.feature_columns.extend([
            'momentum_strength', 'intraday_strength', 'price_position',
            'return_3min', 'return_5min', 'return_10min', 'vol_expansion',
            'sma_5_10_cross', 'sma_10_20_cross', 'rsi_oversold', 'rsi_overbought'
        ])

    def create_all_features(self):
        """
        Create all features and clean the dataframe

        Returns:
            pd.DataFrame: Dataframe with all features
            list: List of feature column names
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        # Create all feature groups
        self.create_price_features()
        self.create_moving_averages()
        self.create_rsi()
        self.create_macd()
        self.create_volatility_features()
        self.create_lag_features()
        self.create_time_features()
        self.create_advanced_features()  # NEW: Advanced high-impact features

        # Remove rows with NaN values (caused by rolling calculations)
        initial_count = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        dropped_rows = initial_count - len(self.df)

        print(f"\nTotal features created: {len(self.feature_columns)}")
        print(f"Features: {', '.join(self.feature_columns[:5])}... (showing first 5)")
        print(f"Dropped {dropped_rows} rows with NaN values (from rolling calculations)")
        print(f"Final dataset: {len(self.df)} rows")
        print("="*60 + "\n")

        return self.df, self.feature_columns

    def get_feature_matrix(self, feature_columns=None):
        """
        Get feature matrix (X) and target vector (y)

        Args:
            feature_columns (list, optional): List of feature columns to use

        Returns:
            tuple: (X, y) - Feature matrix and target vector
        """
        if feature_columns is None:
            feature_columns = self.feature_columns

        X = self.df[feature_columns]
        y = self.df['target']

        return X, y
