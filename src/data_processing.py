# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import VarianceThreshold


class DataPreprocessor:
    """Class for initial data cleaning and preprocessing"""

    def __init__(self, remove_duplicates=True, remove_constant_cols=True):
        self.remove_duplicates = remove_duplicates
        self.remove_constant_cols = remove_constant_cols
        self.constant_columns_ = None
        self.retained_columns_ = None

    def fit(self, X, y=None):
        if self.remove_constant_cols:
            self.constant_columns_ = [
                col for col in X.columns if X[col].nunique(dropna=False) <= 1
            ]
            self.retained_columns_ = [col for col in X.columns if col not in self.constant_columns_]
        else:
            self.retained_columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()

        if self.remove_duplicates:
            X = X.drop_duplicates()

        if self.remove_constant_cols and self.constant_columns_:
            X = X.drop(columns=self.constant_columns_)

        print("Data preprocessing complete.")
        print(f"Retained columns: {len(self.retained_columns_)}")
        return X

    def get_feature_names_out(self, input_features=None):
        return self.retained_columns_


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='TransactionStartTime'):
        self.date_column = date_column
        self.date_columns_ = [
            'TransactionHour', 'TransactionDay', 'TransactionMonth',
            'TransactionYear', 'TransactionDayOfWeek',
            'TransactionDayOfYear', 'IsWeekend'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X = X.copy()
            if not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
                X[self.date_column] = pd.to_datetime(X[self.date_column])

            X['TransactionHour'] = X[self.date_column].dt.hour
            X['TransactionDay'] = X[self.date_column].dt.day
            X['TransactionMonth'] = X[self.date_column].dt.month
            X['TransactionYear'] = X[self.date_column].dt.year
            X['TransactionDayOfWeek'] = X[self.date_column].dt.dayofweek
            X['TransactionDayOfYear'] = X[self.date_column].dt.dayofyear
            X['IsWeekend'] = (X[self.date_column].dt.weekday >= 5).astype(int)

            print("Temporal features extracted successfully.")
            print(f"Length of temporal features: {len(self.date_columns_)}")

            return X.drop(columns=[self.date_column])
        except Exception as e:
            raise ValueError(f"Error extracting temporal features: {str(e)}") from e

    def get_feature_names_out(self, input_features=None):
        return self.date_columns_


class GroupLowCardinality(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.categories_ = {}

    def fit(self, X, y=None):
        try:
            for col in X.select_dtypes(include=['object', 'category']).columns:
                value_counts = X[col].value_counts(normalize=True)
                low_cardinality = value_counts[value_counts < self.threshold].index
                self.categories_[col] = set(low_cardinality)

            print(f"Low cardinality categories identified: {self.categories_}")
            return self
        except Exception as e:
            raise ValueError(f"Error fitting GroupLowCardinality: {str(e)}") from e

    def transform(self, X):
        try:
            check_is_fitted(self, 'categories_')
            X = X.copy()
            for col, low_card in self.categories_.items():
                if col in X.columns:
                    X[col] = np.where(X[col].isin(low_card), 'Other', X[col])

            print("Low cardinality features transformed successfully.")
            print(f"Transformed columns: {list(self.categories_.keys())}")
            print(f"Number of transformed columns: {len(self.categories_)}")
            return X
        except Exception as e:
            raise ValueError(f"Error transforming GroupLowCardinality: {str(e)}") from e

    def get_feature_names_out(self, input_features=None):
        return input_features


class AggregateFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, group_by_cols=None, amount_column='Amount'):
        self.group_by_cols = group_by_cols or ['CustomerId', 'AccountId']
        self.amount_column = amount_column
        self.agg_features_ = {}
        self.feature_names_ = []

    def fit(self, X, y=None):
        for group_col in self.group_by_cols:
            agg_df = X.groupby(group_col, observed=False).agg({
                self.amount_column: ['sum', 'mean', 'std', 'count', 'max', 'min']
            })
            agg_df.columns = [
                f'{group_col}_{stat}_{self.amount_column}' for _, stat in agg_df.columns
            ]
            self.feature_names_.extend(agg_df.columns.tolist())
            agg_df.fillna(0, inplace=True)
            self.agg_features_[group_col] = agg_df
        return self

    def transform(self, X):
        try:
            check_is_fitted(self, 'agg_features_')
            X = X.copy()
            for group_col, agg_df in self.agg_features_.items():
                X = X.merge(agg_df, how='left', left_on=group_col, right_index=True)
                X[agg_df.columns] = X[agg_df.columns].fillna(0)
            print("Aggregate features generated successfully.")
            print(f"Generated features: {self.feature_names_}")
            print(f"Number of aggregate features: {len(self.feature_names_)}")
            return X
        except Exception as e:
            raise ValueError(f"Error transforming AggregateFeatureGenerator: {str(e)}") from e

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


class TypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, dtype_dict):
        self.dtype_dict = dtype_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X = X.copy()
            for col, dtype in self.dtype_dict.items():
                if col in X.columns:
                    X[col] = X[col].astype(dtype)
            print("Data types converted successfully.")
            return X
        except Exception as e:
            raise ValueError(f"Error converting types: {str(e)}") from e

    def get_feature_names_out(self, input_features=None):
        return input_features


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        self.retained_columns_ = None

    def fit(self, X, y=None):
        self.retained_columns_ = [col for col in X.columns if col not in self.columns_to_drop]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        return self.retained_columns_


def create_data_pipeline():
    """Creates the complete data processing pipeline"""
    """
    Feature Types:
    * High Cardinality: TransactionId, BatchId (will be dropped)
    * Medium Cardinality: AccountId, SubscriptionId, CustomerId (used for aggregation)
    * Low Cardinality: CurrencyCode, ProviderId, ProductId, ProductCategory, ChannelId
    """
    
    # Define columns by type
    numeric_features = ['Amount', 'Value', 'PricingStrategy']
    
    # Low cardinality features for one-hot encoding
    low_cardinality_categorical = [
        'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId'
    ]
    
    # Medium cardinality features (used for aggregation)
    aggregation_features = ['AccountId', 'SubscriptionId', 'CustomerId']
    
    # Pipeline for numeric features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('variance_threshold', VarianceThreshold(threshold=0.001))
    ])
    
    # Pipeline for low cardinality categorical features (with one-hot encoding)
    low_cardinality_transformer = Pipeline([
        ('group_low_card', GroupLowCardinality(threshold=0.05)),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('variance_threshold', VarianceThreshold(threshold=0.01))
    ])
    
    # Pipeline for temporal feature extraction
    temporal_transformer = Pipeline([
        ('feature_extractor', FeatureExtractor()),
        ('type_converter', TypeConverter({
            'TransactionHour': 'int32',
            'TransactionDay': 'int32',
            'TransactionMonth': 'int32',
            'TransactionYear': 'int32',
            'TransactionDayOfWeek': 'int32',
            'TransactionDayOfYear': 'int32',
            'IsWeekend': 'int32'
        }))
    ])
    
    # Pipeline for aggregate features
    aggregate_transformer = Pipeline([
        ('aggregator', AggregateFeatureGenerator(group_by_cols=aggregation_features)),
    ])
    
    # Main feature processing pipeline
    feature_processor = FeatureUnion([
        ('numeric', ColumnTransformer([
            ('numeric', numeric_transformer, numeric_features)
        ], remainder='drop')),
        ('categorical', ColumnTransformer([
            ('categorical', low_cardinality_transformer, low_cardinality_categorical)
        ], remainder='drop')),
        ('temporal', temporal_transformer),
        ('aggregate', aggregate_transformer)
    ])
    
    # Complete pipeline
    pipeline = Pipeline([
        ('initial_cleaner', DataPreprocessor()),
        ('drop_high_cardinality', ColumnDropper(columns_to_drop=['TransactionId', 'BatchId'])),
        ('type_converter', TypeConverter({
            'PricingStrategy': 'float32',
            'ProviderId': 'category',
            'ProductId': 'category',
            'ProductCategory': 'category',
            'ChannelId': 'category',
            'AccountId': 'category',
            'SubscriptionId': 'category',
            'CustomerId': 'category'
        })),
        ('feature_processor', feature_processor),
    ])
    
    return pipeline


def get_feature_names(pipeline):
    try:
        feature_processor = pipeline.named_steps['feature_processor']
        feature_names = []
        for name, transformer in feature_processor.transformer_list:
            try:
                if isinstance(transformer, ColumnTransformer):
                    ct_names = transformer.get_feature_names_out()
                    if ct_names is not None:
                        feature_names.extend(ct_names)
                elif isinstance(transformer, Pipeline):
                    last_step = transformer.steps[-1][1]
                    if hasattr(last_step, 'get_feature_names_out'):
                        names = last_step.get_feature_names_out()
                        if names is not None:
                            feature_names.extend(names)
                elif hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out()
                    if names is not None:
                        feature_names.extend(names)
            except Exception as inner_e:
                print(f"Warning extracting names from {name}: {inner_e}")
        return feature_names
    except Exception as e:
        print(f"Failed to extract feature names: {e}")
        return [f"feature_{i}" for i in range(pipeline.named_steps['feature_processor'].transform(X.head(1)).shape[1])]



if __name__ == "__main__":
    try:
        raw_data = pd.read_csv('data/raw/data.csv')
        
        pipeline = create_data_pipeline()  # This builds your full pipeline
        X = raw_data.drop(columns=['FraudResult'], errors='ignore')
        y = raw_data['FraudResult'] if 'FraudResult' in raw_data.columns else None

        X_processed = pipeline.fit_transform(X, y)
        print("Pipeline fitted and data transformed successfully.")
        feature_names = get_feature_names(pipeline)
        
        print(f"Feature names after processing: {feature_names}")
        print(f"Processed data shape: {X_processed.shape}")
        print(f"Number of features after processing: {len(feature_names)}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise
