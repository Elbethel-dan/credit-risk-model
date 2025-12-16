# preprocessing.py
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
# 1. CUSTOMER-LEVEL AGGREGATION WITH COLUMN PRESERVATION
# -------------------------------------------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregate transaction features while preserving
    important categorical and numerical columns.
    """

    def __init__(self, customer_id_col, amount_col, 
                 categorical_cols=None, numeric_cols=None):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.column_mappings_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Prepare aggregation dictionary
        agg_dict = {}
        
        # Amount column aggregations
        agg_dict[self.amount_col] = ['sum', 'mean', 'count', 'std']
        
        # Categorical columns - take the most frequent (mode)
        for col in self.categorical_cols:
            if col in X.columns and col != self.customer_id_col:
                agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan
        
        # Numeric columns (excluding amount) - take mean
        for col in self.numeric_cols:
            if col in X.columns and col != self.amount_col and col != self.customer_id_col:
                agg_dict[col] = 'mean'
        
        # Perform aggregation
        result = X.groupby(self.customer_id_col).agg(agg_dict)
        
        # Flatten multi-level columns
        result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
        result = result.reset_index()
        
        # Rename columns for clarity
        column_rename = {
            f'{self.amount_col}_sum': 'total_transaction_amount',
            f'{self.amount_col}_mean': 'average_transaction_amount',
            f'{self.amount_col}_count': 'transaction_count',
            f'{self.amount_col}_std': 'std_transaction_amount'
        }
        
        result = result.rename(columns=column_rename)
        
        # Store column mappings for reference
        self.column_mappings_ = {
            'customer_id_col': self.customer_id_col,
            'amount_col': self.amount_col,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'result_columns': result.columns.tolist()
        }
        
        return result

    def get_feature_names(self):
        """Get the names of features after transformation."""
        if hasattr(self, 'column_mappings_') and self.column_mappings_:
            return self.column_mappings_['result_columns']
        return None


# -------------------------------------------------------------
# 2. DATE FEATURE EXTRACTION
# -------------------------------------------------------------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, year from datetime column.
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
        
        # Extract features
        X['transaction_hour'] = X[self.date_col].dt.hour
        X['transaction_day'] = X[self.date_col].dt.day
        X['transaction_month'] = X[self.date_col].dt.month
        X['transaction_year'] = X[self.date_col].dt.year
        X['transaction_dayofweek'] = X[self.date_col].dt.dayofweek
        X['transaction_quarter'] = X[self.date_col].dt.quarter
        
        # Drop original date column
        X = X.drop(columns=[self.date_col])
        
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        return X

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 3. OUTLIER REMOVER
# -------------------------------------------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes outliers using the IQR method for numerical features.
    """

    def __init__(self, numeric_cols=None, factor=1.5):
        self.numeric_cols = numeric_cols
        self.factor = factor
        self.bounds_ = {}
        self.feature_names_ = None

    def fit(self, X, y=None):
        if self.numeric_cols is None:
            # Auto-detect numeric columns
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.numeric_cols:
            if col in X.columns:
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
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                mask = (X[col] >= lower) & (X[col] <= upper)
                X = X[mask]
        
        return X

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 4. MISSING VALUE HANDLER
# -------------------------------------------------------------
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handles missing values in the dataset.
    """

    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_impute_values_ = {}
        self.categorical_impute_values_ = {}
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.feature_names_ = X.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Calculate imputation values
        for col in numeric_cols:
            if self.numeric_strategy == 'mean':
                self.numeric_impute_values_[col] = X[col].mean()
            elif self.numeric_strategy == 'median':
                self.numeric_impute_values_[col] = X[col].median()
            elif self.numeric_strategy == 'mode':
                self.numeric_impute_values_[col] = X[col].mode()[0] if not X[col].mode().empty else 0
        
        for col in categorical_cols:
            if self.categorical_strategy == 'most_frequent':
                self.categorical_impute_values_[col] = X[col].mode()[0] if not X[col].mode().empty else 'missing'
            elif self.categorical_strategy == 'constant':
                self.categorical_impute_values_[col] = 'missing'
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Impute numeric columns
        for col, value in self.numeric_impute_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        
        # Impute categorical columns
        for col, value in self.categorical_impute_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        
        return X

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 5. FEATURE ENGINEERING WITH WOE AND IV
# -------------------------------------------------------------
class WoEFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) transformation
    and computes Information Value (IV).
    """

    def __init__(self, iv_threshold=0.02, max_bins=5, min_bin_pct=0.05):
        self.iv_threshold = iv_threshold
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.woe_ = None
        self.feature_names_ = None

    def fit(self, X, y):
        self.woe_ = WOE(
            iv_threshold=self.iv_threshold,
            max_bins=self.max_bins,
            min_bin_pct=self.min_bin_pct
        )
        self.woe_.fit(X, y)
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        transformed = self.woe_.transform(X)
        self.feature_names_ = transformed.columns.tolist()
        return transformed

    def get_feature_names(self):
        return self.feature_names_

    def get_iv_summary(self):
        """
        Returns Information Value (IV) for each feature.
        """
        if hasattr(self.woe_, 'iv_df_'):
            return self.woe_.iv_df_
        return None


# -------------------------------------------------------------
# 6. BUILD FEATURE ENGINEERING PIPELINE
# -------------------------------------------------------------
def build_feature_engineering_pipeline(
    customer_id_col='CustomerId',
    amount_col='Amount',
    date_col='TransactionStartTime',
    target_col=None,
    numeric_cols=None,
    categorical_cols=None,
    scaling_method='standard',
    use_woe=False,
    remove_outliers=True,
    handle_missing=True
):
    """
    Builds a complete feature engineering pipeline.
    
    Parameters:
    -----------
    customer_id_col : str, column name for customer identifier
    amount_col : str, column name for transaction amount
    date_col : str, column name for transaction date/time
    target_col : str, column name for target variable (required for WoE)
    numeric_cols : list, original numeric columns to preserve
    categorical_cols : list, original categorical columns to preserve
    scaling_method : str, 'standard' or 'minmax'
    use_woe : bool, whether to apply WoE transformation
    remove_outliers : bool, whether to remove outliers
    handle_missing : bool, whether to handle missing values
    
    Returns:
    --------
    pipeline : sklearn Pipeline object
    """
    
    # Default column lists if not provided
    if numeric_cols is None:
        numeric_cols = []
    if categorical_cols is None:
        categorical_cols = []
    
    # Initialize pipeline steps
    steps = []
    
    # Step 1: Extract date features
    steps.append(('date_features', DateFeatureExtractor(date_col)))
    
    # Step 2: Aggregate features (preserving specified columns)
    steps.append(('aggregate_features', AggregateFeatures(
        customer_id_col=customer_id_col,
        amount_col=amount_col,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols
    )))
    
    # Step 3: Handle missing values (if requested)
    if handle_missing:
        steps.append(('missing_values', MissingValueHandler()))
    
    # Step 4: Remove outliers (if requested)
    if remove_outliers:
        steps.append(('outlier_removal', OutlierRemover()))
    
    # Step 5: Define which features go to ColumnTransformer
    # These will be determined dynamically after aggregation
    # For now, we'll use a custom transformer that adapts to the data
    
    # Step 6: WoE transformation (if requested and target is available)
    if use_woe:
        if target_col is None:
            raise ValueError("target_col must be specified when use_woe=True")
        steps.append(('woe', WoEFeatureEngineer()))
    
    # Create pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


def get_column_names_after_pipeline(pipeline, X):
    """
    Get the column names after each step of the pipeline.
    
    Parameters:
    -----------
    pipeline : fitted Pipeline object
    X : original DataFrame
    
    Returns:
    --------
    column_history : dict with column names after each step
    """
    column_history = {}
    
    # Get column names after each step
    X_transformed = X.copy()
    for step_name, transformer in pipeline.steps:
        if hasattr(transformer, 'transform'):
            X_transformed = transformer.transform(X_transformed)
            if hasattr(transformer, 'get_feature_names'):
                column_history[step_name] = transformer.get_feature_names()
            else:
                column_history[step_name] = X_transformed.columns.tolist()
    
    return column_history


# -------------------------------------------------------------
# 7. PREPROCESSING PIPELINE FOR MODELING
# -------------------------------------------------------------
def build_modeling_pipeline(
    numeric_features=None,
    categorical_features=None,
    scaling_method='standard',
    use_feature_selection=False
):
    """
    Builds a preprocessing pipeline specifically for modeling.
    To be used AFTER feature engineering.
    """
    
    transformers = []
    
    # Numeric preprocessing
    if numeric_features:
        scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        transformers.append(('num', scaler, numeric_features))
    
    # Categorical preprocessing
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ), categorical_features))
    
    # Create ColumnTransformer
    if transformers:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Keep other columns
        )
        
        steps = [('preprocessor', preprocessor)]
        
        # Add feature selection if requested
        if use_feature_selection:
            from sklearn.feature_selection import SelectKBest, f_classif
            steps.append(('feature_selection', SelectKBest(score_func=f_classif, k=20)))
        
        return Pipeline(steps)
    
    return None