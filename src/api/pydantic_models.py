# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    CustomerId: int
    num__Amount_sum: float
    num__Amount_mean: float
    num__Amount_count: float
    num__Amount_std: Optional[float]
    num__Amount_min: float
    num__Amount_max: float
    num__Amount_median: float
    num__Value_mean: float
    num__Value_std: Optional[float]
    num__Value_min: float
    num__Value_max: float
    num__Value_sum: float
    num__PricingStrategy_mean: float
    num__PricingStrategy_std: Optional[float]
    num__PricingStrategy_min: float
    num__PricingStrategy_max: float
    num__PricingStrategy_sum: float
    num__FraudResult_mean: float
    num__FraudResult_std: Optional[float]
    num__FraudResult_min: float
    num__FraudResult_max: float
    num__FraudResult_sum: float
    num__Amount_avg_value: float
    num__Amount_avg_per_transaction: float
    num__Amount_cv: float
    num__Value_cv: float
    num__PricingStrategy_cv: float
    num__FraudResult_cv: float

    cat__ProviderId_lambda_ProviderId_2: int
    cat__ProviderId_lambda_ProviderId_3: int
    cat__ProviderId_lambda_ProviderId_4: int
    cat__ProviderId_lambda_ProviderId_5: int
    cat__ProviderId_lambda_ProviderId_6: int 
    cat__ProductId_lambda_ProductId_10: int
    cat__ProductId_lambda_ProductId_11: int
    cat__ProductId_lambda_ProductId_12: int
    cat__ProductId_lambda_ProductId_13: int
    cat__ProductId_lambda_ProductId_14: int
    cat__ProductId_lambda_ProductId_15: int
    cat__ProductId_lambda_ProductId_16: int
    cat__ProductId_lambda_ProductId_19: int
    cat__ProductId_lambda_ProductId_2: int
    cat__ProductId_lambda_ProductId_20: int
    cat__ProductId_lambda_ProductId_21: int
    cat__ProductId_lambda_ProductId_22: int
    cat__ProductId_lambda_ProductId_24: int
    cat__ProductId_lambda_ProductId_27: int
    cat__ProductId_lambda_ProductId_3: int
    cat__ProductId_lambda_ProductId_4: int
    cat__ProductId_lambda_ProductId_5: int
    cat__ProductId_lambda_ProductId_6: int
    cat__ProductId_lambda_ProductId_7: int
    cat__ProductId_lambda_ProductId_8: int
    cat__ProductId_lambda_ProductId_9: int

    cat__ProductCategory_lambda_data_bundles: int
    cat__ProductCategory_lambda_financial_services: int
    cat__ProductCategory_lambda_movies: int
    cat__ProductCategory_lambda_other: int
    cat__ProductCategory_lambda_ticket: int
    cat__ProductCategory_lambda_transport: int
    cat__ProductCategory_lambda_tv: int
    cat__ProductCategory_lambda_utility_bill: int

    cat__ChannelId_lambda_ChannelId_2: int
    cat__ChannelId_lambda_ChannelId_3: int
    cat__ChannelId_lambda_ChannelId_5: int


class RiskPredictionResponse(BaseModel):
    CustomerId: int
    risk_probability: float
