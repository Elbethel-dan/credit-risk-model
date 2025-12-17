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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler
)
from xverse.transformer import WOE


# -------------------------------------------------------------
# 1. DYNAMIC PREPROCESSOR
# -------------------------------------------------------------
class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """
    Dynamic preprocessor that adapts to column types and returns DataFrame.
    """
    
    def __init__(self, scaling_method='standard', id_cols=None):
        self.scaling_method = scaling_method
        self.id_cols = id_cols or []
        self.preprocessor_ = None
        self.feature_names_ = None

        
    def fit(self, X, y=None):
    # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Remove ID columns from feature lists
        for col in self.id_cols:
            if col in numeric_features:
                numeric_features.remove(col)
            if col in categorical_features:
                categorical_features.remove(col)

        transformers = []

        if numeric_features:
            scaler = StandardScaler() if self.scaling_method == 'standard' else MinMaxScaler()
            transformers.append(('num', scaler, numeric_features))

        if categorical_features:
            transformers.append(('cat', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first'
            ), categorical_features))

        self.numeric_features_ = numeric_features
        self.categorical_features_ = categorical_features

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        self.preprocessor_.fit(X)
        self.feature_names_ = self.preprocessor_.get_feature_names_out()

        return self

    
    def transform(self, X):
        X = X.copy()

        # Preserve IDs
        ids = X[self.id_cols] if self.id_cols else None
        X_features = X.drop(columns=self.id_cols, errors='ignore')

        transformed = self.preprocessor_.transform(X_features)

        features_df = pd.DataFrame(
            transformed,
            columns=self.feature_names_,
            index=X.index
        )

        # Concatenate IDs back
        if ids is not None:
            return pd.concat([ids.reset_index(drop=True),
                            features_df.reset_index(drop=True)], axis=1)

        return features_df


# -------------------------------------------------------------
# 2. DATAFRAME WRAPPER (Moved outside function)
# -------------------------------------------------------------
class DataFrameWrapper(BaseEstimator, TransformerMixin):
    """
    Ensures the output is a DataFrame.
    """
    
    def __init__(self, steps=None):
        self.steps = steps or []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure output is DataFrame
        if isinstance(X, np.ndarray):
            # Try to get feature names from previous steps
            feature_names = None
            
            # Look for feature names in previous steps
            if self.steps:
                for step_name, transformer in reversed(self.steps):
                    if hasattr(transformer, 'get_feature_names'):
                        names = transformer.get_feature_names()
                        if names is not None:
                            feature_names = names
                            break
            
            if feature_names and len(feature_names) == X.shape[1]:
                return pd.DataFrame(X, columns=feature_names)
            else:
                return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X


# -------------------------------------------------------------
# 3. CUSTOMER AGGREGATION FEATURES
# -------------------------------------------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregate features from transaction data.
    """

    def __init__(self, customer_id_col, amount_col, 
                 categorical_cols=None, numeric_cols=None):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Prepare aggregation dictionary
        agg_dict = {}
        
        # 1. Amount statistics
        agg_dict[self.amount_col] = ['sum', 'mean', 'count', 'std', 'min', 'max', 'median']
        
        # 2. Categorical columns - take mode (most frequent)
        for col in self.categorical_cols:
            if col in X.columns and col != self.customer_id_col:
                agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan
        
        # 3. Other numeric columns - statistics
        for col in self.numeric_cols:
            if col in X.columns and col != self.amount_col and col != self.customer_id_col:
                agg_dict[col] = ['mean', 'std', 'min', 'max', 'sum']
        
        # Perform aggregation
        result = X.groupby(self.customer_id_col).agg(agg_dict)
        
        # Flatten multi-level columns
        result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
        
        # **Clean <lambda> from categorical columns**
        result.columns = [col.replace('<lambda>', '').strip('_') for col in result.columns]
        
        result = result.reset_index()
        
        # Create derived features
        if f'{self.amount_col}_sum' in result.columns and f'{self.amount_col}_count' in result.columns:
            result[f'{self.amount_col}_avg_value'] = (
                result[f'{self.amount_col}_sum'] / result[f'{self.amount_col}_count']
            )
        
        # Handle infinite values
        result = result.replace([np.inf, -np.inf], np.nan)
        
        self.feature_names_ = result.columns.tolist()
        
        return result 


    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 4. DATE FEATURE EXTRACTION
# -------------------------------------------------------------
class ExtractDateFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts features from datetime columns.
    """

    def __init__(self, date_col):
        self.date_col = date_col
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Convert to datetime
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')
        
        # Extract basic date features
        X['transaction_hour'] = X[self.date_col].dt.hour
        X['transaction_day'] = X[self.date_col].dt.day
        X['transaction_month'] = X[self.date_col].dt.month
        X['transaction_year'] = X[self.date_col].dt.year
        X['transaction_dayofweek'] = X[self.date_col].dt.dayofweek
        X['transaction_quarter'] = X[self.date_col].dt.quarter
        
        # Create derived features
        X['is_weekend'] = X['transaction_dayofweek'].isin([5, 6]).astype(int)
        X['is_business_hours'] = ((X['transaction_hour'] >= 9) & (X['transaction_hour'] <= 17)).astype(int)
        
        # Time period features
        X['is_morning'] = ((X['transaction_hour'] >= 6) & (X['transaction_hour'] < 12)).astype(int)
        X['is_afternoon'] = ((X['transaction_hour'] >= 12) & (X['transaction_hour'] < 18)).astype(int)
        X['is_evening'] = ((X['transaction_hour'] >= 18) & (X['transaction_hour'] < 24)).astype(int)
        X['is_night'] = ((X['transaction_hour'] >= 0) & (X['transaction_hour'] < 6)).astype(int)
        
        # Drop original date column
        X = X.drop(columns=[self.date_col])
        
        self.feature_names_ = X.columns.tolist()
        
        return X

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 5. OUTLIER REMOVER (IQR Method)
# -------------------------------------------------------------
class RemoveOutliers(BaseEstimator, TransformerMixin):
    """
    Removes outliers using IQR method.
    """

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
            
            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            
            self.bounds_[col] = (lower, upper)
        
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        
        # Create a mask for rows to keep
        keep_mask = pd.Series(True, index=X.index)
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                col_mask = (X[col] >= lower) & (X[col] <= upper)
                keep_mask = keep_mask & col_mask
        
        # Remove outliers
        X_clean = X[keep_mask].copy()
        
        removed_count = len(X) - len(X_clean)
        if removed_count > 0:
            print(f"  Outlier removal: {removed_count} rows removed "
                  f"({(removed_count / len(X) * 100):.1f}%)")
        
        return X_clean

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 6. FEATURE ENGINEERING
# -------------------------------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Creates additional engineered features.
    """

    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Amount-related features
        amount_cols = [col for col in X.columns if 'Amount' in col and 'sum' in col]
        for amount_col in amount_cols:
            col_prefix = amount_col.replace('_sum', '')
            
            # Create ratio features if we have count
            count_col = amount_col.replace('sum', 'count')
            if count_col in X.columns:
                X[f'{col_prefix}_avg_per_transaction'] = X[amount_col] / X[count_col]
        
        # Time pattern features
        time_cols = [col for col in X.columns if any(time in col for time in ['hour', 'day', 'month', 'year'])]
        for time_col in time_cols:
            if '_std' in time_col:
                base_col = time_col.replace('_std', '')
                mean_col = f'{base_col}_mean'
                if mean_col in X.columns:
                    X[f'{base_col}_consistency'] = 1 / (X[time_col] + 1)
        
        # Behavioral features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if '_std' in col:
                mean_col = col.replace('_std', '_mean')
                if mean_col in numeric_cols:
                    X[f'{col.replace("_std", "")}_cv'] = X[col] / (X[mean_col] + 1e-10)
        
        # Fill any NaN values
        X = X.fillna(0)
        
        self.feature_names_ = X.columns.tolist()
        
        return X

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 7. WOE TRANSFORMER (Weight of Evidence & Information Value)
# -------------------------------------------------------------
class WoEFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) transformation and computes IV.
    """

    def __init__(self, target_col=None, iv_threshold=0.02, max_bins=5, min_bin_pct=0.05):
        self.target_col = target_col
        self.iv_threshold = iv_threshold
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.woe_ = None
        self.feature_names_ = None
        self.iv_summary_ = None

    def fit(self, X, y=None):
        # If target is provided separately
        if y is not None:
            target = y
        # If target column is specified in X
        elif self.target_col and self.target_col in X.columns:
            target = X[self.target_col]
            X = X.drop(columns=[self.target_col])
        # No target available
        else:
            print("⚠️  No target provided for WoE transformation. Skipping WoE.")
            self.woe_ = None
            self.feature_names_ = X.columns.tolist()
            return self
        
        # Initialize and fit WoE transformer
        self.woe_ = WOE(
            iv_threshold=self.iv_threshold,
            max_bins=self.max_bins,
            min_bin_pct=self.min_bin_pct
        )
        
        try:
            self.woe_.fit(X, target)
            self.iv_summary_ = self.woe_.iv_df_ if hasattr(self.woe_, 'iv_df_') else None
            self.feature_names_ = X.columns.tolist()
            print(f"✅ WoE fitting successful.")
        except Exception as e:
            print(f"⚠️  WoE fitting failed: {e}. Skipping WoE.")
            self.woe_ = None
            self.feature_names_ = X.columns.tolist()
        
        return self

    def transform(self, X):
        if self.woe_ is None:
            return X
        
        try:
            # If target column exists in X, remove it
            if self.target_col and self.target_col in X.columns:
                X = X.drop(columns=[self.target_col])
            
            transformed = self.woe_.transform(X)
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
# 8. MAIN FEATURE ENGINEERING PIPELINE BUILDER
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
    """
    Builds complete feature engineering pipeline.
    """
    
    if categorical_cols is None:
        categorical_cols = []
    if numeric_cols is None:
        numeric_cols = []
    
    # Build the pipeline steps
    steps = []
    
    # Step 1: Extract date features
    steps.append(('extract_date_features', ExtractDateFeatures(date_col)))
    
    # Step 2: Customer aggregation
    steps.append(('aggregate_features', AggregateFeatures(
        customer_id_col=customer_id_col,
        amount_col=amount_col,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols
    )))
    
    # Step 3: Feature engineering
    steps.append(('feature_engineering', FeatureEngineering()))
    
    # Step 4: Remove outliers (optional)
    if remove_outliers:
        steps.append(('remove_outliers', RemoveOutliers()))
    
    # Step 5: Create preprocessing for encoding and scaling
    steps.append(('preprocessing', DynamicPreprocessor(scaling_method=scaling_method, id_cols=[customer_id_col]   # ✅ KEEP ID FOR MERGING
)))

    # Step 6: WoE transformation (optional)
    if use_woe:
        steps.append(('woe_transformer', WoEFeatureTransformer(target_col=woe_target_col)))
    
    # Step 7: Ensure DataFrame output
    # Pass the steps to DataFrameWrapper so it can look for feature names
    steps.append(('dataframe_wrapper', DataFrameWrapper(steps=steps[:-1])))
    
    return Pipeline(steps)


# -------------------------------------------------------------
# 9. UTILITY FUNCTIONS
# -------------------------------------------------------------
def save_pipeline(pipeline, filepath):
    """
    Save pipeline to disk.
    """
    import joblib
    joblib.dump(pipeline, filepath)
    print(f"✅ Pipeline saved to: {filepath}")


def load_pipeline(filepath):
    """
    Load pipeline from disk.
    """
    import joblib
    return joblib.load(filepath)