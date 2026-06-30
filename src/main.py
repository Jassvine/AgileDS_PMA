import pandas as pd
import numpy as np

def load_data(path='/content/train.csv'):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame({'val': [1, 2, 3]})

def validate_data(df):
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    
    # Identify numerical columns
    num_df = df.select_dtypes(include=[np.number])
    
    # Exclude latitude/longitude from the positive check as they can be negative
    cols_to_check = [c for c in num_df.columns if 'latitude' not in c.lower() and 'longitude' not in c.lower()]
    
    positive_check = True
    failing_cols = []
    
    for col in cols_to_check:
        if not (num_df[col] >= 0).all():
            positive_check = False
            failing_cols.append(col)
            
    return {
        'missing_values': int(missing_count),
        'duplicates': int(duplicate_count),
        'positive_values': positive_check,
        'failing_columns': failing_cols
    }
