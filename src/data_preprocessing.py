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
# 1. DATE FEATURE EXTRACTION
# -------------------------------------------------------------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts date features from datetime column.
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
# 2. CUSTOMER AGGREGATION FEATURES
# -------------------------------------------------------------
class CustomerAggregator(BaseEstimator, TransformerMixin):
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
        
        # 1. Basic amount statistics
        agg_dict[self.amount_col] = ['sum', 'mean', 'count', 'std', 'min', 'max', 'median']
        
        # 2. Time between transactions (if date features exist)
        date_cols = ['transaction_hour', 'transaction_day', 'transaction_month', 
                    'transaction_year', 'transaction_dayofweek']
        for col in date_cols:
            if col in X.columns:
                agg_dict[col] = ['mean', 'std', 'min', 'max']
        
        # 3. Categorical columns - mode (most frequent)
        for col in self.categorical_cols:
            if col in X.columns and col != self.customer_id_col:
                agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan
        
        # 4. Other numeric columns - statistics
        for col in self.numeric_cols:
            if col in X.columns and col != self.amount_col and col != self.customer_id_col:
                agg_dict[col] = ['mean', 'std', 'min', 'max', 'sum']
        
        # Perform aggregation
        result = X.groupby(self.customer_id_col).agg(agg_dict)
        
        # Flatten multi-level columns
        result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
        result = result.reset_index()
        
        # Create derived features
        if f'{self.amount_col}_sum' in result.columns and f'{self.amount_col}_count' in result.columns:
            # Average transaction value
            result[f'{self.amount_col}_avg_value'] = result[f'{self.amount_col}_sum'] / result[f'{self.amount_col}_count']
            
            # Transaction frequency features
            if 'transaction_day_count' in result.columns:
                result['days_between_transactions'] = result['transaction_day_count'] / result[f'{self.amount_col}_count']
        
        # Handle infinite values
        result = result.replace([np.inf, -np.inf], np.nan)
        
        self.feature_names_ = result.columns.tolist()
        
        return result

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
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
                
                # Transaction frequency
                if 'transaction_day_std' in X.columns:
                    X[f'{col_prefix}_frequency_variability'] = X[count_col] / (X['transaction_day_std'] + 1)
        
        # Time pattern features
        if 'transaction_hour_std' in X.columns:
            X['transaction_time_consistency'] = 1 / (X['transaction_hour_std'] + 1)
        
        if 'transaction_dayofweek_std' in X.columns:
            X['day_pattern_consistency'] = 1 / (X['transaction_dayofweek_std'] + 1)
        
        # Behavioral features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if '_std' in col:
                mean_col = col.replace('_std', '_mean')
                if mean_col in numeric_cols:
                    X[f'{col.replace("_std", "")}_cv'] = X[col] / (X[mean_col] + 1e-10)  # Coefficient of variation
        
        # Fill NaN values
        X = X.fillna(0)
        
        self.feature_names_ = X.columns.tolist()
        
        return X

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 4. OUTLIER REMOVER (IQR Method)
# -------------------------------------------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes outliers using IQR method.
    """

    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}
        self.feature_names_ = None
        self.original_shape_ = None

    def fit(self, X, y=None):
        self.original_shape_ = X.shape
        
        # Get numeric columns
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
        
        print(f"  Outlier removal: {len(X) - len(X_clean)} rows removed "
              f"({((len(X) - len(X_clean)) / len(X) * 100):.1f}%)")
        
        return X_clean

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 5. MISSING VALUE IMPUTER
# -------------------------------------------------------------
class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values.
    """

    def __init__(self, numeric_strategy='median', categorical_strategy='most_frequent'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_values_ = {}
        self.categorical_values_ = {}
        self.feature_names_ = None

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Numeric imputation values
        for col in numeric_cols:
            if self.numeric_strategy == 'mean':
                self.numeric_values_[col] = X[col].mean()
            elif self.numeric_strategy == 'median':
                self.numeric_values_[col] = X[col].median()
            elif self.numeric_strategy == 'zero':
                self.numeric_values_[col] = 
        
        # Categorical imputation values
        for col in categorical_cols:
            if self.categorical_strategy == 'most_frequent':
                self.categorical_values_[col] = X[col].mode()[0] if not X[col].mode().empty else 'missing'
            elif self.categorical_strategy == 'constant':
                self.categorical_values_[col] = 'missing'
        
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        
        # Count missing values before imputation
        missing_before = X.isnull().sum().sum()
        
        # Impute numeric columns
        for col, value in self.numeric_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        
        # Impute categorical columns
        for col, value in self.categorical_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        
        # Count missing values after imputation
        missing_after = X.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"  Missing value imputation: {missing_before} values imputed")
        
        return X

    def get_feature_names(self):
        return self.feature_names_


# -------------------------------------------------------------
# 6. WOE TRANSFORMER (Weight of Evidence)
# -------------------------------------------------------------
class WoEFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) transformation.
    Can be used with or without target.
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
            
            print(f"  WoE transformation applied. IV summary available.")
            return transformed
        except Exception as e:
            print(f"⚠️  WoE transformation failed: {e}. Returning original features.")
            return X

    def get_feature_names(self):
        return self.feature_names_

    def get_iv_summary(self):
        """
        Returns Information Value (IV) summary if available.
        """
        return self.iv_summary_


# -------------------------------------------------------------
# 7. MAIN FEATURE ENGINEERING PIPELINE
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
    handle_missing=True
):
    """
    Builds complete feature engineering pipeline.
    
    Parameters:
    -----------
    customer_id_col : str, column name for customer ID
    amount_col : str, column name for transaction amount
    date_col : str, column name for transaction date
    categorical_cols : list, categorical columns to preserve
    numeric_cols : list, numeric columns to preserve
    use_woe : bool, whether to apply WoE transformation
    woe_target_col : str, target column name for WoE (optional)
    remove_outliers : bool, whether to remove outliers
    handle_missing : bool, whether to handle missing values
    
    Returns:
    --------
    pipeline : sklearn Pipeline object
    """
    
    if categorical_cols is None:
        categorical_cols = []
    if numeric_cols is None:
        numeric_cols = []
    
    # Build the pipeline steps
    steps = []
    
    # Step 1: Extract date features
    steps.append(('date_extractor', DateFeatureExtractor(date_col)))
    
    # Step 2: Customer aggregation
    steps.append(('customer_aggregator', CustomerAggregator(
        customer_id_col=customer_id_col,
        amount_col=amount_col,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols
    )))
    
    # Step 3: Feature engineering
    steps.append(('feature_engineer', FeatureEngineer()))
    
    # Step 4: Handle missing values
    if handle_missing:
        steps.append(('missing_imputer', MissingValueImputer()))
    
    # Step 5: Remove outliers
    if remove_outliers:
        steps.append(('outlier_remover', OutlierRemover()))
    
    # Step 6: WoE transformation (optional)
    if use_woe:
        steps.append(('woe_transformer', WoEFeatureEngineer(target_col=woe_target_col)))
    
    return Pipeline(steps)


# -------------------------------------------------------------
# 8. MODEL PREPROCESSING PIPELINE
# -------------------------------------------------------------
def build_model_preprocessing_pipeline(
    numeric_features=None,
    categorical_features=None,
    scaling_method='standard',
    reduce_dimensions=False,
    n_components=None
):
    """
    Builds preprocessing pipeline for modeling.
    """
    
    transformers = []
    
    # Numeric preprocessing
    if numeric_features and len(numeric_features) > 0:
        scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        transformers.append(('num', scaler, numeric_features))
    
    # Categorical preprocessing
    if categorical_features and len(categorical_features) > 0:
        transformers.append(('cat', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop='first'
        ), categorical_features))
    
    # Create ColumnTransformer
    if transformers:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        steps = [('preprocessor', preprocessor)]
        
        # Optional dimensionality reduction
        if reduce_dimensions:
            from sklearn.decomposition import PCA
            if n_components is None:
                steps.append(('pca', PCA(n_components=0.95)))  # Keep 95% variance
            else:
                steps.append(('pca', PCA(n_components=n_components)))
        
        return Pipeline(steps)
    
    return None


# -------------------------------------------------------------
# 9. UTILITY FUNCTIONS
# -------------------------------------------------------------
def get_feature_summary(X):
    """
    Get summary of engineered features.
    """
    summary = {
        'total_features': X.shape[1],
        'numeric_features': X.select_dtypes(include=[np.number]).shape[1],
        'categorical_features': X.select_dtypes(exclude=[np.number]).shape[1],
        'feature_names': X.columns.tolist(),
        'missing_values': X.isnull().sum().sum(),
        'data_shape': X.shape
    }
    return summary


def analyze_feature_importance(X, y=None, method='correlation'):
    """
    Analyze feature importance using different methods.
    
    Parameters:
    -----------
    X : DataFrame, features
    y : Series, target (optional)
    method : str, 'correlation', 'variance', or 'mutual_info'
    
    Returns:
    --------
    importance_df : DataFrame with feature importance scores
    """
    
    importance_df = pd.DataFrame({'feature': X.columns})
    
    if method == 'correlation' and y is not None:
        # Correlation with target
        correlations = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                corr = np.abs(X[col].corr(y))
                correlations.append(corr if not pd.isna(corr) else 0)
            else:
                correlations.append(0)
        importance_df['importance'] = correlations
        importance_df['method'] = 'correlation'
    
    elif method == 'variance':
        # Feature variance
        variances = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                variances.append(X[col].var())
            else:
                variances.append(0)
        importance_df['importance'] = variances
        importance_df['method'] = 'variance'
    
    elif method == 'mutual_info' and y is not None:
        # Mutual information with target
        from sklearn.feature_selection import mutual_info_classif
        try:
            X_numeric = X.select_dtypes(include=[np.number])
            mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
            
            # Create full importance array
            full_scores = np.zeros(len(X.columns))
            numeric_idx = 0
            for i, col in enumerate(X.columns):
                if col in X_numeric.columns:
                    full_scores[i] = mi_scores[numeric_idx]
                    numeric_idx += 1
            
            importance_df['importance'] = full_scores
            importance_df['method'] = 'mutual_info'
        except:
            importance_df['importance'] = 0
            importance_df['method'] = 'mutual_info_failed'
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df