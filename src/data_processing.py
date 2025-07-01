# src/data_processing.py
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional


class DataPreprocessor:
    """Class for comprehensive data cleaning, preprocessing, and feature engineering"""
    
    def __init__(self, data: pd.DataFrame, date_column: str = "TransactionStartTime"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data: Input DataFrame to preprocess
            date_column: Name of the datetime column (default "TransactionStartTime")
        """
        self.data = data.copy()  # Work with a copy to avoid modifying original
        self.date_column = date_column
        self._preprocessed = False
        self._report = {}  # Store preprocessing statistics
        
    def get_data(self) -> pd.DataFrame:
        """Return the preprocessed data."""
        if not self._preprocessed:
            print("Warning: Data hasn't been preprocessed yet. Call preprocess() first.")
        return self.data
        
    def drop_columns(self, columns: List[str]) -> None:
        """
        Drop specified columns from the DataFrame.
        
        Args:
            columns: List of column names to drop
        """
        existing_columns = [col for col in columns if col in self.data.columns]
        missing_columns = set(columns) - set(existing_columns)
        
        if missing_columns:
            print(f"Warning: Columns not found: {missing_columns}")
            
        try:
            self.data.drop(columns=existing_columns, inplace=True)
            self._report['dropped_columns'] = existing_columns
            print(f"Dropped columns: {existing_columns}")
        except Exception as e:
            print(f"Error dropping columns: {e}")
    
    def rename_columns(self, column_mapping: Dict[str, str]) -> None:
        """
        Rename columns in the DataFrame based on a mapping.
        
        Args:
            column_mapping: Dictionary of {old_name: new_name} pairs
        """
        existing_mapping = {k: v for k, v in column_mapping.items() 
                           if k in self.data.columns}
        missing_keys = set(column_mapping.keys()) - set(existing_mapping.keys())
        
        if missing_keys:
            print(f"Warning: Columns not found: {missing_keys}")
            
        try:
            self.data.rename(columns=existing_mapping, inplace=True)
            self._report['renamed_columns'] = existing_mapping
            print(f"Renamed columns: {existing_mapping}")
        except Exception as e:
            print(f"Error renaming columns: {e}")
            
    def group_by_column(self, group_by_column: str, agg_funcs: Dict[str, Union[str, List[str]]]) -> Optional[pd.DataFrame]:
        """
        Group the DataFrame by a specified column and apply aggregation functions.
        
        Args:
            group_by_column: Column name to group by
            agg_funcs: Dictionary of {column: aggregation_function} pairs
            
        Returns:
            Grouped DataFrame or None if error occurs
        """
        if group_by_column not in self.data.columns:
            print(f"Column '{group_by_column}' not found in DataFrame.")
            return None
            
        valid_cols = {k: v for k, v in agg_funcs.items() if k in self.data.columns}
        if len(valid_cols) != len(agg_funcs):
            missing = set(agg_funcs.keys()) - set(valid_cols.keys())
            print(f"Warning: Columns not found: {missing}")
            
        try:
            grouped_data = self.data.groupby(group_by_column).agg(valid_cols)
            return grouped_data
        except Exception as e:
            print(f"Error grouping by column '{group_by_column}': {e}")
            return None
    
    def _get_high_cardinality_cols(self, threshold: float = 0.8) -> List[str]:
        """Identify columns with high cardinality (many unique values)."""
        high_card_cols = []
        for col in self.data.select_dtypes(include=['object', 'category']).columns:

            unique_ratio = self.data[col].nunique() / len(self.data)
            if unique_ratio > threshold:
                high_card_cols.append(col)
        self._report['high_cardinality_cols'] = high_card_cols
        return high_card_cols
    
    def _get_constant_columns(self) -> List[str]:
        """Identify columns with constant or quasi-constant values."""
        constant_cols = []
        for col in self.data.columns:
            if self.data[col].nunique() <= 1:
                constant_cols.append(col)
        self._report['constant_columns'] = constant_cols
        return constant_cols
    
    def _get_low_cardinality_cols(self, threshold: int = 35) -> List[str]:
        """Identify columns with low cardinality (few unique values)."""
        low_card_cols = []
        for col in self.data.select_dtypes(include=['object', 'category']).columns:
            if self.data[col].nunique() <= threshold:
                low_card_cols.append(col)
        self._report['low_cardinality_cols'] = low_card_cols
        return low_card_cols
    
    def replace_low_frequency_classes(self, columns: List[str], threshold: float = 0.05, 
                                    replacement: str = "Other") -> None:
        """
        Replace low-frequency classes with specified replacement value.
        
        Args:
            columns: List of column names to process
            threshold: Frequency threshold (default 0.05)
            replacement: Value to use for replacement (default "Other")
        """
        replacement_stats = {}
        
        for col in columns:
            if col not in self.data.columns:
                print(f"Column '{col}' not found in DataFrame. Skipping.")
                continue
                
            if not pd.api.types.is_categorical_dtype(self.data[col]) and not pd.api.types.is_object_dtype(self.data[col]):
                print(f"Column '{col}' is not categorical. Skipping.")
                continue
                
            value_counts = self.data[col].value_counts(normalize=True)
            low_freq_classes = value_counts[value_counts < threshold].index.tolist()
            
            if low_freq_classes:
                replace_dict = {cls: replacement for cls in low_freq_classes}
                self.data[col] = self.data[col].replace(replace_dict)
                replacement_stats[col] = {
                    'num_replaced': len(low_freq_classes),
                    'replaced_classes': low_freq_classes
                }
                print(f"Replaced {len(low_freq_classes)} classes in '{col}' with '{replacement}'")
            else:
                print(f"No low-frequency classes found in '{col}' (threshold={threshold})")
                
        self._report['low_frequency_replacements'] = replacement_stats
    
    def create_aggregate_features(self, group_by_column: str, agg_columns: Dict[str, List[str]]) -> None:
        """
        Create aggregate features grouped by a specified column.

        Args:
            group_by_column: Column to group by (e.g., 'CustomerId')
            agg_columns: Dictionary where keys are aggregation types (e.g., 'mean', 'sum')
                        and values are lists of columns to apply them on.
        """
        if group_by_column not in self.data.columns:
            print(f"Grouping column '{group_by_column}' not found.")
            return

        # Transform to pandas expected format: {column: [agg1, agg2, ...]}
        agg_columns_transformed = {}
        for col, metrics in agg_columns.items():
            for metric in metrics:
                if col in self.data.columns:
                    agg_columns_transformed.setdefault(col, []).append(metric)

        if not agg_columns_transformed:
            print("No valid aggregation columns specified")
            return

        try:
            grouped = self.data.groupby(group_by_column).agg(agg_columns_transformed)

            # Flatten column names: Amount_sum, Amount_mean, etc.
            grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
            grouped.reset_index(inplace=True)

            self.data = self.data.merge(grouped, on=group_by_column, how='left')

            self._report['aggregate_features'] = list(grouped.columns)
            print(f"Created aggregate features: {list(grouped.columns)}")

        except Exception as e:
            print(f"Error creating aggregate features: {e}")

    def extract_datetime_features(self, datetime_col: Optional[str] = None) -> None:
        """
        Extract features from datetime column (e.g., hour, day, month).
        
        Args:
            datetime_col: Name of datetime column (uses class default if None)
        """
        col = datetime_col or self.date_column
        if col not in self.data.columns:
            print(f"Datetime column '{col}' not found.")
            return
            
        try:
            # Basic datetime features
            self.data[f'{col}_year'] = self.data[col].dt.year
            self.data[f'{col}_month'] = self.data[col].dt.month
            self.data[f'{col}_day'] = self.data[col].dt.day
            self.data[f'{col}_hour'] = self.data[col].dt.hour
            self.data[f'{col}_minute'] = self.data[col].dt.minute
            self.data[f'{col}_dayofweek'] = self.data[col].dt.dayofweek
            self.data[f'{col}_is_weekend'] = self.data[col].dt.dayofweek >= 5
            
            # Time since reference features
            ref_date = self.data[col].max()
            self.data[f'{col}_days_since'] = (ref_date - self.data[col]).dt.days
            self.data[f'{col}_weeks_since'] = self.data[f'{col}_days_since'] // 7
            
            # Cyclical encoding for periodic features
            self.data[f'{col}_hour_sin'] = np.sin(2 * np.pi * self.data[col].dt.hour/24)
            self.data[f'{col}_hour_cos'] = np.cos(2 * np.pi * self.data[col].dt.hour/24)
            self.data[f'{col}_day_sin'] = np.sin(2 * np.pi * self.data[col].dt.day/31)
            self.data[f'{col}_day_cos'] = np.cos(2 * np.pi * self.data[col].dt.day/31)
            
            # Add to report
            new_features = [
                f'{col}_year', f'{col}_month', f'{col}_day',
                f'{col}_hour', f'{col}_minute', f'{col}_dayofweek',
                f'{col}_is_weekend', f'{col}_days_since', f'{col}_weeks_since',
                f'{col}_hour_sin', f'{col}_hour_cos', f'{col}_day_sin', f'{col}_day_cos'
            ]
            self._report['extracted_datetime_features'] = new_features
            print(f"Extracted datetime features from '{col}'")
            
        except Exception as e:
            print(f"Error extracting datetime features: {e}")


    def preprocess(self) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        try:
            # 1. Handle datetime
            if self.date_column in self.data.columns:
                self.data[self.date_column] = pd.to_datetime(
                    self.data[self.date_column], errors='coerce'
                )
                self.data.dropna(subset=[self.date_column], inplace=True)
                self.extract_datetime_features()
            
            # 2. Handle missing values
            num_cols = self.data.select_dtypes(include=np.number).columns
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
            
            # Numerical: median imputation
            if not num_cols.empty:
                self.data[num_cols] = self.data[num_cols].fillna(self.data[num_cols].median())
            
            # Categorical: mode imputation
            if not cat_cols.empty:
                for col in cat_cols:
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
            
            # 3. Handle low-frequency categories
            low_card_cols = self._get_low_cardinality_cols()
            if low_card_cols:
                self.replace_low_frequency_classes(low_card_cols)
            
            # 4. Remove constant columns
            constant_cols = self._get_constant_columns()
            if constant_cols:
                self.drop_columns(constant_cols)
            
            # 5. Create aggregate features (example)
            print("*********************************************************************************************")
            print("Creating aggregate features...")
            if 'CustomerId' in self.data.columns and 'Amount' in self.data.columns:
                self.create_aggregate_features(
                    group_by_column='CustomerId',
                    agg_columns={
                        'Amount': ['sum', 'mean', 'count', 'std', 'median', 'min', 'max']
                    }
                )
                print("Aggregate features created for 'CustomerId' and 'Amount'.")
            
            self._preprocessed = True
            print("Preprocessing completed successfully.")
            return self.data
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise
            
    def get_preprocessing_report(self) -> Dict:
        """Return a dictionary with preprocessing statistics and changes."""
        return self._report.copy()



# Example usage
if __name__ == "__main__":
    # Sample data
    df = pd.read_csv("data/raw/data.csv")
    
    # Initialize and preprocess
    preprocessor = DataPreprocessor(df)
    preprocessor.preprocess()
    
    
    dropped_cols = ["BatchId", "AccountId", "SubscriptionId"]
    
    # Drop unnecessary columns
    preprocessor.drop_columns(dropped_cols)
    
    # Get processed data
    processed_df = preprocessor.get_data()
    print("\nProcessed Data:")
    print(processed_df.head())
    
    processed_df.to_csv("data/processed/processed_data.csv", index=False)
    print("\nProcessed data saved to 'data/processed/processed_data.csv'")
    
    
    # Get report
    report = preprocessor.get_preprocessing_report()
    print("\nPreprocessing Report:")
    
    for key, value in report.items():
        print(f"{key}: {value}")
