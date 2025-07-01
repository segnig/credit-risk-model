from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def create_feature_encoding_pipeline(categorical_features, numerical_features):
    """
    Create a ColumnTransformer for encoding and scaling features.

    Parameters:
    categorical_features (list): List of categorical feature names.
    numerical_features (list): List of numerical feature names.

    Returns:
    ColumnTransformer: A transformer that applies encoders/scalers to appropriate columns.
    """
    transformers = []

    if categorical_features:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        )

    if numerical_features:
        transformers.append(
            ('num', StandardScaler(), numerical_features)
        )

    column_transformer = ColumnTransformer(transformers=transformers)
    return column_transformer

def preprocess_data(data, categorical_features, numerical_features):
    """
    Preprocess the data using the feature encoding pipeline.

    Parameters:
    data (DataFrame): The input data to preprocess.
    categorical_features (list): List of categorical feature names.
    numerical_features (list): List of numerical feature names.

    Returns:
    DataFrame: The preprocessed data.
    """
    pipeline = create_feature_encoding_pipeline(categorical_features, numerical_features)
    processed_data = pipeline.fit_transform(data)

    # Generate column names
    cat_features_encoded = []
    if categorical_features:
        ohe = pipeline.named_transformers_['cat']
        cat_features_encoded = ohe.get_feature_names_out(categorical_features)

    feature_names = list(cat_features_encoded) + numerical_features
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    processed_df["TransactionStartTime"] = data["TransactionStartTime"]
    processed_df["CustomerId"] = data["CustomerId"]

    return processed_df

# Define feature sets
cat_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
numeric_features = [
    'Amount', 'Value', 'PricingStrategy', 'FraudResult', 'TransactionStartTime_year', 'TransactionStartTime_month',
    'TransactionStartTime_day', 'TransactionStartTime_hour', 'TransactionStartTime_minute', 'TransactionStartTime_dayofweek',
    'TransactionStartTime_is_weekend', 'TransactionStartTime_days_since', 'TransactionStartTime_weeks_since', 'TransactionStartTime_hour_sin',
    'TransactionStartTime_hour_cos', 'TransactionStartTime_day_sin', 'TransactionStartTime_day_cos', 'Amount_sum', 'Amount_mean',
    'Amount_count', 'Amount_std', 'Amount_median', 'Amount_min', 'Amount_max'
]

# Load data
FILE_PATH = "data/processed/processed_data.csv"
data = pd.read_csv(FILE_PATH)

# Preprocess
processed_df = preprocess_data(data, cat_features, numeric_features)

# Save processed data
PROCESSED_FILE_PATH = "data/processed/processed_data_encoded.csv"
processed_df.to_csv(PROCESSED_FILE_PATH, index=False)