"""
Probability of Default (PD) Models

This module implements various approaches for modeling Probability of Default (PD)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class PDModelBase:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:

        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of default"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary default outcome"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class LogisticPDModel(PDModelBase):
    """
    Logistic Regression model for PD estimation

    Based on the standard approach in credit risk where:
    logit(PD) = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ

    Where PD = 1 / (1 + exp(-logit(PD)))
    """

    def __init__(self, regularization: str = 'l2', C: float = 1.0):
        super().__init__("Logistic_PD")
        self.regularization = regularization
        self.C = C
        self.model = LogisticRegression(
            penalty=regularization,
            C=C,
            solver='liblinear',
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit logistic regression model"""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict PD probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients with interpretation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_[0],
            'Odds_Ratio': np.exp(self.model.coef_[0])
        })

        # Add intercept
        intercept_row = pd.DataFrame({
            'Feature': ['Intercept'],
            'Coefficient': [self.model.intercept_[0]],
            'Odds_Ratio': [np.exp(self.model.intercept_[0])]
        })

        return pd.concat([intercept_row, coef_df], ignore_index=True)


class RFPDModel(PDModelBase):
    """Random Forest model for PD estimation"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest_PD")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Random Forest model"""
        self.feature_names = X.columns.tolist()
        # Random Forest doesn't require scaling
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict PD probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return importance_df


class GBMPDModel(PDModelBase):
    """Gradient Boosting Machine model for PD estimation"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        super().__init__("GBM_PD")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit GBM model"""
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict PD probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class PDModelEvaluator:
    """Evaluation metrics and tools for PD models"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)

        metrics = {
            'AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'Accuracy': np.mean(y_true == y_pred),
            'Precision': np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0,
            'Recall': np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        }

        return metrics

    @staticmethod
    def gini_coefficient(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Gini coefficient (2*AUC - 1)"""
        auc = roc_auc_score(y_true, y_pred_proba)
        return 2 * auc - 1

    @staticmethod
    def ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        # Sort by predicted probability
        sorted_idx = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true[sorted_idx]

        # Calculate cumulative distributions
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        cum_pos = np.cumsum(y_true_sorted) / n_pos
        cum_neg = np.cumsum(1 - y_true_sorted) / n_neg

        # KS statistic is maximum difference
        ks_stat = np.max(np.abs(cum_pos - cum_neg))
        return ks_stat


class PDModelFactory:
    """Factory class for creating different PD models"""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> PDModelBase:
        """Create PD model based on type"""
        if model_type.lower() == 'logistic':
            return LogisticPDModel(**kwargs)
        elif model_type.lower() == 'random_forest':
            return RFPDModel(**kwargs)
        elif model_type.lower() == 'gbm':
            return GBMPDModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Utility functions for data generation and preprocessing
def generate_synthetic_credit_data(n_samples: int = 1000,
                                   default_rate: float = 0.1,
                                   random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic credit data for testing"""
    np.random.seed(random_state)

    # Generate features
    data = {
        'credit_score': np.random.normal(650, 100, n_samples),
        'debt_to_income': np.random.lognormal(np.log(0.3), 0.5, n_samples),
        'loan_amount': np.random.lognormal(np.log(50000), 0.8, n_samples),
        'employment_years': np.random.exponential(5, n_samples),
        'age': np.random.normal(40, 12, n_samples),
        'num_credit_lines': np.random.poisson(4, n_samples),
        'utilization_rate': np.random.beta(2, 5, n_samples)
    }

    df = pd.DataFrame(data)

    # Clip unrealistic values
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    df['debt_to_income'] = np.clip(df['debt_to_income'], 0, 2)
    df['age'] = np.clip(df['age'], 18, 80)
    df['utilization_rate'] = np.clip(df['utilization_rate'], 0, 1)

    # Generate default indicator based on risk factors
    risk_score = (
            -0.01 * (df['credit_score'] - 500) +
            2.0 * df['debt_to_income'] +
            0.5 * df['utilization_rate'] +
            -0.1 * df['employment_years'] +
            -0.02 * (df['age'] - 30)
    )

    # Convert to probability and generate defaults
    default_prob = 1 / (1 + np.exp(-risk_score))
    df['default'] = np.random.binomial(1, default_prob, n_samples)

    # Adjust default rate to match target
    current_rate = df['default'].mean()
    if current_rate != default_rate:
        # Simple adjustment - randomly flip some outcomes
        n_adjust = int(abs(current_rate - default_rate) * n_samples)
        if current_rate > default_rate:
            # Reduce defaults
            default_idx = df[df['default'] == 1].index
            flip_idx = np.random.choice(default_idx, n_adjust, replace=False)
            df.loc[flip_idx, 'default'] = 0
        else:
            # Increase defaults
            non_default_idx = df[df['default'] == 0].index
            flip_idx = np.random.choice(non_default_idx, n_adjust, replace=False)
            df.loc[flip_idx, 'default'] = 1

    return df

