# preprocessing.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)

from xverse.transformer import WOE



# -------------------------------------------------------------
# 1. CUSTOMER-LEVEL AGGREGATION
# -------------------------------------------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregate transaction features.
    """

    def __init__(self, customer_id_col, amount_col):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = (
            X.groupby(self.customer_id_col)[self.amount_col]
            .agg(
                total_transaction_amount='sum',
                average_transaction_amount='mean',
                transaction_count='count',
                std_transaction_amount='std'
            )
            .reset_index()
        )

        return agg_df


# -------------------------------------------------------------
# 2. DATE FEATURE EXTRACTION
# -------------------------------------------------------------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, year from datetime column.
    """

    def __init__(self, date_col):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])

        X['transaction_hour'] = X[self.date_col].dt.hour
        X['transaction_day'] = X[self.date_col].dt.day
        X['transaction_month'] = X[self.date_col].dt.month
        X['transaction_year'] = X[self.date_col].dt.year

        return X.drop(columns=[self.date_col])
# -------------------------------------------------------------
# 3. Removing Outliers
# -------------------------------------------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes outliers using the IQR method for numerical features.
    """

    def __init__(self, numeric_cols, factor=1.5):
        self.numeric_cols = numeric_cols
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        for col in self.numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR

            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        X = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            X = X[(X[col] >= lower) & (X[col] <= upper)]

        return X


# -------------------------------------------------------------
# 4. Feature Engineering with WoE and IV
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

    def fit(self, X, y):
        self.woe_ = WOE(
            iv_threshold=self.iv_threshold,
            max_bins=self.max_bins,
            min_bin_pct=self.min_bin_pct
        )
        self.woe_.fit(X, y)
        return self

    def transform(self, X):
        return self.woe_.transform(X)

    def get_iv_summary(self):
        """
        Returns Information Value (IV) for each feature.
        """
        return self.woe_.iv_df_


# -------------------------------------------------------------
# 5. build_feature_engineering_pipeline
# -------------------------------------------------------------
def build_feature_engineering_pipeline(
    numeric_cols,
    categorical_cols,
    customer_id_col,
    amount_col,
    date_col,
    scaling_method='standard',
    use_woe=False,
    remove_outliers=True
):
    """
    Builds a full Task 3 feature engineering pipeline.
    """
    
    # First create the aggregate features, then handle outliers
    steps = [
        ('date_features', DateFeatureExtractor(date_col)),
        ('aggregate_features', AggregateFeatures(customer_id_col, amount_col))
    ]
    
    # Outlier removal should come AFTER aggregation
    if remove_outliers:
        steps.append(('outlier_removal', OutlierRemover(numeric_cols)))
    
    # Then do preprocessing
    scaler = (
        StandardScaler() if scaling_method == 'standard'
        else MinMaxScaler()
    )
    
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )
    
    steps.append(('preprocessing', preprocessing))
    
    # WoE if needed
    if use_woe:
        steps.append(('woe', WoEFeatureEngineer()))
    
    return Pipeline(steps)
