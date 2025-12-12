import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------
# 1. CUSTOMER-LEVEL AGGREGATION
# -------------------------------------------------------------
class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data into customer-level features.
    """
    def __init__(self, customer_id='SubscriptionId', amount_col='Amount', value_col='Value'):
        self.customer_id = customer_id
        self.amount_col = amount_col
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df[self.amount_col] = pd.to_numeric(df[self.amount_col], errors='coerce')
        df[self.value_col] = pd.to_numeric(df[self.value_col], errors='coerce')

        group = df.groupby(self.customer_id)

        agg = pd.DataFrame({
            "SubscriptionId": group.size().index,
            "total_amount": group[self.amount_col].sum().values,
            "avg_amount": group[self.amount_col].mean().values,
            "std_amount": group[self.amount_col].std().fillna(0).values,
            "count_tx": group.size().values,
            "total_value": group[self.value_col].sum().values,
            "avg_value": group[self.value_col].mean().values,
        })

        return agg.fillna(0)


# -------------------------------------------------------------
# 2. DATE FEATURE EXTRACTION
# -------------------------------------------------------------
class DateFeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Extracts date features (hour/day/month/weekdays) and aggregates per customer.
    """
    def __init__(self, customer_id="SubscriptionId", date_col="TransactionDate"):
        self.customer_id = customer_id
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')

        df["tx_hour"] = df[self.date_col].dt.hour
        df["tx_day"] = df[self.date_col].dt.day
        df["tx_month"] = df[self.date_col].dt.month
        df["tx_weekday"] = df[self.date_col].dt.weekday

        grp = df.groupby(self.customer_id)

        agg = pd.DataFrame({
            "SubscriptionId": grp.size().index,
            "mean_hour": grp["tx_hour"].mean().values,
            "unique_days": grp["tx_day"].nunique().values,
            "unique_months": grp["tx_month"].nunique().values,
            "weekday_var": grp["tx_weekday"].var().fillna(0).values,
        })

        return agg.fillna(0)


# -------------------------------------------------------------
# 3. WEIGHT OF EVIDENCE (WOE) + INFORMATION VALUE (IV)
# -------------------------------------------------------------
class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Computes WOE and transforms categorical variables.
    """
    def __init__(self, categorical_cols=None, replace_missing="MISSING"):
        self.categorical_cols = categorical_cols
        self.replace_missing = replace_missing
        self.woe_map_ = {}
        self.iv_dict_ = {}

    def _calculate_woe_iv(self, feature, y):
        df = pd.DataFrame({"x": feature.fillna(self.replace_missing), "y": y})
        g = df.groupby("x").agg(bad=("y", "sum"), total=("y", "count"))
        g["good"] = g["total"] - g["bad"]

        # Avoid division by zero
        g["bad_rate"] = (g["bad"] + 0.5) / g["bad"].sum()
        g["good_rate"] = (g["good"] + 0.5) / g["good"].sum()

        g["woe"] = np.log(g["good_rate"] / g["bad_rate"])
        g["iv"] = (g["good_rate"] - g["bad_rate"]) * g["woe"]

        return g["woe"].to_dict(), g["iv"].sum()

    def fit(self, X, y):
        if self.categorical_cols is None:
            self.categorical_cols = X.columns.tolist()

        for col in self.categorical_cols:
            mapping, iv = self._calculate_woe_iv(X[col], y)
            self.woe_map_[col] = mapping
            self.iv_dict_[col] = iv

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.categorical_cols:
            X[col] = (
                X[col].fillna(self.replace_missing)
                .map(self.woe_map_[col])
                .fillna(0.0)
            )
        return X


# -------------------------------------------------------------
# 4. BUILD FULL FEATURESET (MERGING EVERYTHING)
# -------------------------------------------------------------
def build_customer_features(df):
    """
    Combines:
    - Customer aggregates
    - Date feature aggregates
    - Category mode per customer
    Returns a customer-level dataframe.
    """

    agg1 = CustomerAggregator().transform(df)
    agg2 = DateFeatureAggregator().transform(df)

    customer_df = agg1.merge(agg2, on="SubscriptionId", how="left")

    # Gather categorical mode (ProviderId, ProductId, etc.)
    categorical_cols = ["ProviderId", "ProductId", "ProductCategory",
                        "ChannelId", "PricingStrategy"]

    for col in categorical_cols:
        mode_series = (
            df.groupby("SubscriptionId")[col]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else np.nan)
            .reset_index()
        )
        customer_df = customer_df.merge(mode_series, on="SubscriptionId", how="left")

    return customer_df.fillna(0)


# -------------------------------------------------------------
# 5. BUILD FULL PIPELINE FOR MODELING
# -------------------------------------------------------------
def build_model_pipeline(numeric_features, categorical_features):
    """
    Builds an sklearn Pipeline using:
    - WoE for categorical columns
    - Standard scaling for numeric columns
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features),  # WOE already numeric
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    return model_pipeline


# -------------------------------------------------------------
# 6. HELPER TO APPLY WOE + PIPELINE TO CUSTOMER DF
# -------------------------------------------------------------
def apply_woe(customer_df, target_col, categorical_cols):
    woe = WOETransformer(categorical_cols=categorical_cols)
    transformed = woe.fit_transform(customer_df[categorical_cols], customer_df[target_col])

    # insert back into customer_df
    df_out = customer_df.copy()
    for col in categorical_cols:
        df_out[col] = transformed[col]

    return df_out, woe
