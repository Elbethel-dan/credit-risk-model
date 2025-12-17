# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from src.data_preprocessing import AggregateFeatures, ExtractDateFeatures


# -------------------------------
# 1. Test AggregateFeatures
# -------------------------------
def test_aggregate_features():
    # Sample transaction data
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 300],
        'Category': ['A', 'A', 'B']
    })

    agg = AggregateFeatures(customer_id_col='CustomerId', amount_col='Amount', categorical_cols=['Category'])
    result = agg.fit(df).transform(df)
    
    # Check expected columns
    expected_cols = [
        'CustomerId', 'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std',
        'Amount_min', 'Amount_max', 'Amount_median', 'Category'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Column {col} missing in aggregation result"
    
    # Check aggregation values
    assert result.loc[result['CustomerId'] == 1, 'Amount_sum'].values[0] == 300
    assert result.loc[result['CustomerId'] == 2, 'Amount_count'].values[0] == 1

# -------------------------------
# 2. Test ExtractDateFeatures
# -------------------------------
def test_extract_date_features():
    df = pd.DataFrame({
        'TransactionDate': pd.to_datetime(['2025-12-15 10:00', '2025-12-16 20:00'])
    })
    
    extractor = ExtractDateFeatures(date_col='TransactionDate')
    result = extractor.fit(df).transform(df)
    
    # Check new columns
    expected_cols = [
        'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year',
        'transaction_dayofweek', 'transaction_quarter', 'is_weekend', 'is_business_hours',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Column {col} missing in date features"

    # Check specific feature values
    assert result.loc[0, 'transaction_hour'] == 10
    assert result.loc[1, 'is_evening'] == 1
