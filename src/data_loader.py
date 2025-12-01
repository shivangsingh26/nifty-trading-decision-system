"""
Data Loader Module
Handles loading, cleaning, and preparing the NIFTY dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    """
    Loads and preprocesses NIFTY intraday data
    """

    def __init__(self, data_path):
        """
        Initialize DataLoader with path to CSV file

        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """
        Load CSV file and perform basic preprocessing

        Returns:
            pd.DataFrame: Loaded and preprocessed dataframe
        """
        print("Loading data from CSV...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")

        return self.df

    def preprocess(self):
        """
        Preprocess the data:
        - Convert timestamp to datetime
        - Sort by timestamp ascending (chronological order)
        - Remove duplicates
        - Handle missing values

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Preprocessing data...")

        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Sort by timestamp in ascending order (oldest first)
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        print(f"Data sorted chronologically: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")

        # Remove duplicates based on timestamp
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['timestamp'], keep='first')
        removed_duplicates = initial_count - len(self.df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")

        # Check for missing values in key columns
        key_columns = ['open', 'high', 'low', 'close']
        missing_counts = self.df[key_columns].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values found:\n{missing_counts}")
            print("Dropping rows with missing values in OHLC columns...")
            self.df = self.df.dropna(subset=key_columns).reset_index(drop=True)

        print(f"Preprocessing complete: {len(self.df)} rows remaining")
        return self.df

    def create_target(self):
        """
        Create target column:
        - target = 1 if next candle's close > current close
        - target = 0 if next candle's close <= current close

        Returns:
            pd.DataFrame: Dataframe with target column
        """
        print("Creating target variable...")

        # Shift close price to get next candle's close
        self.df['next_close'] = self.df['close'].shift(-1)

        # Create binary target
        self.df['target'] = (self.df['next_close'] > self.df['close']).astype(int)

        # Drop the last row (no next_close available)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['next_close']).reset_index(drop=True)

        # Check class distribution
        class_distribution = self.df['target'].value_counts()
        print(f"Target variable created. Class distribution:")
        print(f"  0 (Price Down): {class_distribution[0]} ({class_distribution[0]/len(self.df)*100:.2f}%)")
        print(f"  1 (Price Up):   {class_distribution[1]} ({class_distribution[1]/len(self.df)*100:.2f}%)")
        print(f"Dropped last row (no next candle). Remaining: {len(self.df)} rows")

        return self.df

    def get_processed_data(self):
        """
        Execute complete data loading pipeline

        Returns:
            pd.DataFrame: Fully processed dataframe ready for feature engineering
        """
        self.load_data()
        self.preprocess()
        self.create_target()

        return self.df
