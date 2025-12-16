# data_preprocessing.py
"""
Feature Engineering Pipeline for Credit Risk Modeling
Includes: Aggregate Features, Date Extraction, Encoding, Scaling, and WoE/IV
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from xverse.transformer import WOE

# -------------------------------------------------------------
# 1. UTILITY FUNCTION
# -------------------------------------------------------------
def mode_or_nan(x):
    """Return mode or np.nan if empty"""
    m = x.mode()
    return m[0] if not m.empty else np.nan

# -------------------------------------------------------------
# 2. CUSTOMER AGGREGATION FEATURES
# -------------------------------------------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Creates customer-level aggregate features from transaction data."""
    def __init__(self, customer_id_col, amount_col, categorical_cols=None, numeric_cols=None):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg_dict = {}
        agg_dict[self.amount_col] = ['sum', 'mean', 'count', 'std', 'min', 'max', 'median']

        for col in self.categorical_cols:
            if col in X.columns and col != self.customer_id_col:
                agg_dict[col] = mode_or_nan

        for col in self.numeric_cols:
            if col in X.columns and col != self.amount_col and col != self.customer_id_col:
                agg_dict[col] = ['mean', 'std', 'min', 'max', 'sum']

        result = X.groupby(self.customer_id_col).agg(agg_dict)
        result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
        result = result.reset_index()

        if f'{self.amount_col}_sum' in result.columns and f'{self.amount_col}_count' in result.columns:
            result[f'{self.amount_col}_avg_value'] = result[f'{self.amount_col}_sum'] / result[f'{self.amount_col}_count']

        result = result.replace([np.inf, -np.inf], np.nan)
        self.feature_names_ = result.columns.tolist()
        return result

    def get_feature_names(self):
        return self.feature_names_

# -------------------------------------------------------------
# 3. DATE FEATURE EXTRACTION
# -------------------------------------------------------------
class ExtractDateFeatures(BaseEstimator, TransformerMixin):
    """Extracts features from datetime columns."""
    def __init__(self, date_col):
        self.date_col = date_col
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')

        X['transaction_hour'] = X[self.date_col].dt.hour
        X['transaction_day'] = X[self.date_col].dt.day
        X['transaction_month'] = X[self.date_col].dt.month
        X['transaction_year'] = X[self.date_col].dt.year
        X['transaction_dayofweek'] = X[self.date_col].dt.dayofweek
        X['transaction_quarter'] = X[self.date_col].dt.quarter

        X['is_weekend'] = X['transaction_dayofweek'].isin([5, 6]).astype(int)
        X['is_business_hours'] = ((X['transaction_hour'] >= 9) & (X['transaction_hour'] <= 17)).astype(int)
        X['is_morning'] = ((X['transaction_hour'] >= 6) & (X['transaction_hour'] < 12)).astype(int)
        X['is_afternoon'] = ((X['transaction_hour'] >= 12) & (X['transaction_hour'] < 18)).astype(int)
        X['is_evening'] = ((X['transaction_hour'] >= 18) & (X['transaction_hour'] < 24)).astype(int)
        X['is_night'] = ((X['transaction_hour'] >= 0) & (X['transaction_hour'] < 6)).astype(int)

        X = X.drop(columns=[self.date_col])
        self.feature_names_ = X.columns.tolist()
        return X

    def get_feature_names(self):
        return self.feature_names_

# -------------------------------------------------------------
# 4. OUTLIER REMOVER (IQR Method)
# -------------------------------------------------------------
class RemoveOutliers(BaseEstimator, TransformerMixin):
    """Removes outliers using IQR method."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}
        self.feature_names_ = None

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - self.factor * IQR, Q3 + self.factor * IQR)
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        keep_mask = pd.Series(True, index=X.index)
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                keep_mask &= (X[col] >= lower) & (X[col] <= upper)
        X_clean = X[keep_mask].copy()
        removed_count = len(X) - len(X_clean)
        if removed_count > 0:
            print(f"  Outlier removal: {removed_count} rows removed ({(removed_count / len(X) * 100):.1f}%)")
        return X_clean

    def get_feature_names(self):
        return self.feature_names_

# -------------------------------------------------------------
# 5. FEATURE ENGINEERING
# -------------------------------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Creates additional engineered features."""
    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        amount_cols = [col for col in X.columns if 'Amount' in col and 'sum' in col]
        for amount_col in amount_cols:
            count_col = amount_col.replace('sum', 'count')
            if count_col in X.columns:
                X[f'{amount_col.replace("_sum", "")}_avg_per_transaction'] = X[amount_col] / X[count_col]

        time_cols = [col for col in X.columns if any(time in col for time in ['hour', 'day', 'month', 'year'])]
        for time_col in time_cols:
            if '_std' in time_col:
                base_col = time_col.replace('_std', '')
                mean_col = f'{base_col}_mean'
                if mean_col in X.columns:
                    X[f'{base_col}_consistency'] = 1 / (X[time_col] + 1)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if '_std' in col:
                mean_col = col.replace('_std', '_mean')
                if mean_col in numeric_cols:
                    X[f'{col.replace("_std", "")}_cv'] = X[col] / (X[mean_col] + 1e-10)

        X = X.fillna(0)
        self.feature_names_ = X.columns.tolist()
        return X

    def get_feature_names(self):
        return self.feature_names_

# -------------------------------------------------------------
# 6. WOE TRANSFORMER
# -------------------------------------------------------------
class WoEFeatureTransformer(BaseEstimator, TransformerMixin):
    """Applies Weight of Evidence (WoE) transformation and computes IV."""
    def __init__(self, target_col=None, iv_threshold=0.02, max_bins=5, min_bin_pct=0.05):
        self.target_col = target_col
        self.iv_threshold = iv_threshold
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.woe_ = None
        self.feature_names_ = None
        self.iv_summary_ = None

    def fit(self, X, y=None):
        target = y if y is not None else (X[self.target_col] if self.target_col in X.columns else None)
        if target is None:
            print("⚠️  No target provided for WoE. Skipping.")
            self.feature_names_ = X.columns.tolist()
            return self

        X_ = X.drop(columns=[self.target_col]) if self.target_col in X.columns else X
        self.woe_ = WOE(iv_threshold=self.iv_threshold, max_bins=self.max_bins, min_bin_pct=self.min_bin_pct)

        try:
            self.woe_.fit(X_, target)
            self.iv_summary_ = getattr(self.woe_, 'iv_df_', None)
            self.feature_names_ = X_.columns.tolist()
            print(f"✅ WoE fitting successful.")
        except Exception as e:
            print(f"⚠️  WoE fitting failed: {e}. Skipping WoE.")
            self.woe_ = None
            self.feature_names_ = X_.columns.tolist()

        return self

    def transform(self, X):
        if self.woe_ is None:
            return X
        X_ = X.drop(columns=[self.target_col]) if self.target_col in X.columns else X
        try:
            transformed = self.woe_.transform(X_)
            self.feature_names_ = transformed.columns.tolist()
            print(f"✅ WoE transformation applied.")
            return transformed
        except Exception as e:
            print(f"⚠️  WoE transformation failed: {e}. Returning original features.")
            return X

    def get_feature_names(self):
        return self.feature_names_

    def get_iv_summary(self):
        return self.iv_summary_

# -------------------------------------------------------------
# 7. DYNAMIC PREPROCESSOR (MODULE LEVEL!)
# -------------------------------------------------------------
class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """Automatic numeric/categorical scaling and encoding."""
    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        self.preprocessor_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in ['CustomerId', 'customer_id']:
            if col in categorical_features:
                categorical_features.remove(col)

        transformers = []
        if numeric_features:
            scaler = StandardScaler() if self.scaling_method == 'standard' else MinMaxScaler()
            transformers.append(('num', scaler, numeric_features))
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features))

        if transformers:
            self.preprocessor_ = ColumnTransformer(transformers=transformers, remainder='drop')
            self.preprocessor_.fit(X)
            self.feature_names_ = self.preprocessor_.get_feature_names_out().tolist() if hasattr(self.preprocessor_, 'get_feature_names_out') else []

        return self

    def transform(self, X):
        if self.preprocessor_ is None:
            return X
        return self.preprocessor_.transform(X)

    def get_feature_names(self):
        return self.feature_names_

# -------------------------------------------------------------
# 8. PIPELINE BUILDER
# -------------------------------------------------------------
def build_feature_engineering_pipeline(
    customer_id_col='CustomerId',
    amount_col='Amount',
    date_col='TransactionStartTime',
    categorical_cols=None,
    numeric_cols=None,
    use_woe=False,
    woe_target_col=None,
    remove_outliers=True,
    scaling_method='standard'
):
    steps = [
        ('extract_date_features', ExtractDateFeatures(date_col)),
        ('aggregate_features', AggregateFeatures(customer_id_col, amount_col, categorical_cols, numeric_cols)),
        ('feature_engineering', FeatureEngineering())
    ]

    if remove_outliers:
        steps.append(('remove_outliers', RemoveOutliers()))

    steps.append(('preprocessing', DynamicPreprocessor(scaling_method=scaling_method)))

    if use_woe:
        steps.append(('woe_transformer', WoEFeatureTransformer(target_col=woe_target_col)))

    return Pipeline(steps)

# -------------------------------------------------------------
# 9. UTILITY FUNCTIONS
# -------------------------------------------------------------
def get_feature_names_from_pipeline(pipeline, X_sample):
    X_transformed = X_sample.copy()
    feature_names = None
    for _, transformer in pipeline.steps:
        if hasattr(transformer, 'transform'):
            X_transformed = transformer.transform(X_transformed)
            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out()
                if names is not None and len(names) > 0:
                    feature_names = names.tolist()
            elif hasattr(transformer, "get_feature_names"):
                names = transformer.get_feature_names()
                if names is not None and len(names) > 0:
                    feature_names = list(names)
    return feature_names

def analyze_features(X, y=None):
    analysis = {
        'total_features': X.shape[1],
        'numeric_features': X.select_dtypes(include=[np.number]).shape[1],
        'categorical_features': X.select_dtypes(exclude=[np.number]).shape[1],
        'feature_names': X.columns.tolist(),
        'data_shape': X.shape,
        'missing_values': X.isnull().sum().sum()
    }
    numeric_features = X.select_dtypes(include=[np.number])
    if not numeric_features.empty:
        variances = numeric_features.var().sort_values(ascending=False)
        analysis['top_variance_features'] = variances.head(10).to_dict()
        analysis['total_variance'] = variances.sum()
    if y is not None and not numeric_features.empty:
        correlations = {col: abs(X[col].corr(y)) for col in numeric_features.columns if not pd.isna(X[col].corr(y))}
        analysis['top_correlated_features'] = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10])
    return analysis

def save_pipeline(pipeline, filepath):
    import joblib
    joblib.dump(pipeline, filepath)
    print(f"✅ Pipeline saved to: {filepath}")

def load_pipeline(filepath):
    import joblib
    return joblib.load(filepath)
