"""
Model Trainer Module
Trains multiple ML models and compares their performance
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import pickle
import os

class ModelTrainer:
    """
    Trains and compares multiple ML models for binary classification
    """

    def __init__(self, X_train, X_test, y_train, y_test, models_dir):
        """
        Initialize ModelTrainer

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            models_dir: Directory to save trained models
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models_dir = models_dir
        self.models = {}
        self.results = {}

    def train_logistic_regression(self, params, use_scaling=True):
        """
        Train Logistic Regression model with optional feature scaling

        Args:
            params (dict): Model parameters
            use_scaling (bool): Whether to scale features (recommended for LR)

        Returns:
            LogisticRegression: Trained model
        """
        print("\n" + "-"*60)
        print("Training Logistic Regression...")
        print("-"*60)

        if use_scaling:
            print("Scaling features for Logistic Regression...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
        else:
            X_train_scaled = self.X_train
            X_test_scaled = self.X_test

        model = LogisticRegression(**params)
        model.fit(X_train_scaled, self.y_train)

        # Store scaler for predictions
        self.scaler = scaler if use_scaling else None

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def train_random_forest(self, params):
        """
        Train Random Forest model

        Args:
            params (dict): Model parameters

        Returns:
            RandomForestClassifier: Trained model
        """
        print("\n" + "-"*60)
        print("Training Random Forest...")
        print("-"*60)

        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def train_lightgbm(self, params):
        """
        Train LightGBM model

        Args:
            params (dict): Model parameters

        Returns:
            lgb.LGBMClassifier: Trained model
        """
        print("\n" + "-"*60)
        print("Training LightGBM...")
        print("-"*60)

        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        self.models['lightgbm'] = model
        self.results['lightgbm'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def train_xgboost(self, params):
        """
        Train XGBoost model

        Args:
            params (dict): Model parameters

        Returns:
            xgb.XGBClassifier: Trained model
        """
        print("\n" + "-"*60)
        print("Training XGBoost...")
        print("-"*60)

        model = xgb.XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0, average='weighted')
        recall = recall_score(self.y_test, y_pred, zero_division=0, average='weighted')
        f1 = f1_score(self.y_test, y_pred, zero_division=0, average='weighted')

        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def compare_models(self):
        """
        Compare all trained models and select the best one

        Returns:
            tuple: (best_model_name, best_model, comparison_df)
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        print(comparison_df.to_string(index=False))
        print("="*60)

        # Select best model based on accuracy
        best_model_name = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
        best_model = self.models[best_model_name]

        print(f"\nBest Model: {comparison_df.iloc[0]['Model']}")
        print(f"Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        print("="*60 + "\n")

        return best_model_name, best_model, comparison_df

    def save_model(self, model, model_name):
        """
        Save trained model to disk

        Args:
            model: Trained model object
            model_name (str): Name of the model
        """
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved: {model_path}")

    def get_feature_importance(self, model, model_name, feature_names, top_n=15):
        """
        Get feature importance for tree-based models

        Args:
            model: Trained model
            model_name (str): Name of the model
            feature_names (list): List of feature names
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name in ['random_forest', 'lightgbm', 'xgboost']:
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            return importance_df.head(top_n)
        else:
            return None
