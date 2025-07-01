import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_input(data) -> pd.DataFrame:
    df = data

    # Parse datetime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Date features
    df["TransactionStartTime_year"] = df["TransactionStartTime"].dt.year
    df["TransactionStartTime_month"] = df["TransactionStartTime"].dt.month
    df["TransactionStartTime_day"] = df["TransactionStartTime"].dt.day
    df["TransactionStartTime_hour"] = df["TransactionStartTime"].dt.hour
    df["TransactionStartTime_minute"] = df["TransactionStartTime"].dt.minute
    df["TransactionStartTime_dayofweek"] = df["TransactionStartTime"].dt.dayofweek
    df["TransactionStartTime_is_weekend"] = df["TransactionStartTime_dayofweek"].apply(lambda x: int(x >= 5))
    df["TransactionStartTime_days_since"] = (datetime.now() - df["TransactionStartTime"]).dt.days
    df["TransactionStartTime_weeks_since"] = df["TransactionStartTime_days_since"] // 7
    df["TransactionStartTime_hour_sin"] = np.sin(2 * np.pi * df["TransactionStartTime_hour"] / 24)
    df["TransactionStartTime_hour_cos"] = np.cos(2 * np.pi * df["TransactionStartTime_hour"] / 24)
    df["TransactionStartTime_day_sin"] = np.sin(2 * np.pi * df["TransactionStartTime_day"] / 31)
    df["TransactionStartTime_day_cos"] = np.cos(2 * np.pi * df["TransactionStartTime_day"] / 31)

    # One-hot encode manually for expected columns (same as training)
    provider_cols = {
        'ProviderId_ProviderId_1': 0,
        'ProviderId_ProviderId_4': 0,
        'ProviderId_ProviderId_5': 0,
        'ProviderId_ProviderId_6': 0,
        'ProviderId_Other': 0,
    }
    product_cols = {
        'ProductId_ProductId_10': 0,
        'ProductId_ProductId_15': 0,
        'ProductId_ProductId_3': 0,
        'ProductId_ProductId_6': 0,
        'ProductId_Other': 0,
    }
    category_cols = {
        'ProductCategory_airtime': 0,
        'ProductCategory_financial_services': 0,
        'ProductCategory_Other': 0,
    }
    channel_cols = {
        'ChannelId_ChannelId_2': 0,
        'ChannelId_ChannelId_3': 0,
        'ChannelId_Other': 0,
    }

    # Fill one-hot columns
    provider_col = f"ProviderId_{df['ProviderId'].values[0]}"
    product_col = f"ProductId_{df['ProductId'].values[0]}"
    category_col = f"ProductCategory_{df['ProductCategory'].values[0]}"
    channel_col = f"ChannelId_{df['ChannelId'].values[0]}"

    if provider_col in provider_cols:
        provider_cols[provider_col] = 1
    else:
        provider_cols['ProviderId_Other'] = 1

    if product_col in product_cols:
        product_cols[product_col] = 1
    else:
        product_cols['ProductId_Other'] = 1

    if category_col in category_cols:
        category_cols[category_col] = 1
    else:
        category_cols['ProductCategory_Other'] = 1

    if channel_col in channel_cols:
        channel_cols[channel_col] = 1
    else:
        channel_cols['ChannelId_Other'] = 1

    # Combine all features
    final_features = {
        **provider_cols,
        **product_cols,
        **category_cols,
        **channel_cols,
        "Amount": df["Amount"].values[0],
        "Value": df["Value"].values[0],
        "PricingStrategy": df["PricingStrategy"].values[0],
        "FraudResult": df["FraudResult"].values[0],
        "TransactionStartTime_year": df["TransactionStartTime_year"].values[0],
        "TransactionStartTime_month": df["TransactionStartTime_month"].values[0],
        "TransactionStartTime_day": df["TransactionStartTime_day"].values[0],
        "TransactionStartTime_hour": df["TransactionStartTime_hour"].values[0],
        "TransactionStartTime_minute": df["TransactionStartTime_minute"].values[0],
        "TransactionStartTime_dayofweek": df["TransactionStartTime_dayofweek"].values[0],
        "TransactionStartTime_is_weekend": df["TransactionStartTime_is_weekend"].values[0],
        "TransactionStartTime_days_since": df["TransactionStartTime_days_since"].values[0],
        "TransactionStartTime_weeks_since": df["TransactionStartTime_weeks_since"].values[0],
        "TransactionStartTime_hour_sin": df["TransactionStartTime_hour_sin"].values[0],
        "TransactionStartTime_hour_cos": df["TransactionStartTime_hour_cos"].values[0],
        "TransactionStartTime_day_sin": df["TransactionStartTime_day_sin"].values[0],
        "TransactionStartTime_day_cos": df["TransactionStartTime_day_cos"].values[0],
        "Amount_sum": df["Amount"].values[0],  # optional default
        "Amount_mean": df["Amount"].values[0],
        "Amount_count": 1,
        "Amount_std": 0.0,
        "Amount_median": df["Amount"].values[0],
        "Amount_min": df["Amount"].values[0],
        "Amount_max": df["Amount"].values[0]
    }

    return pd.DataFrame([final_features])
