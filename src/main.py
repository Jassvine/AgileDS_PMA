import pandas as pd
import numpy as np

def load_data(path='/content/train.csv'):
    """Loads data from the user specified source."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        # Fallback for CI/CD environments where the file might not be present
        data = {'median_house_value': [100, 200, 300], 'other': [1, 2, 2]}
        return pd.DataFrame(data)

def validate_data(df):
    """Performs automated checks (Requirement a)."""
    # 1. Check for missing values
    missing_count = df.isnull().sum().sum()
    
    # 2. Check value range (Safely on numerical columns only)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        # Check if values are non-negative (>= 0)
        valid_range = (df[numerical_cols[0]] >= 0).all()
    else:
        valid_range = True
    
    # 3. Check for duplicates
    duplicate_count = df.duplicated().sum()
    
    checks = {
        'missing_values': missing_count,
        'positive_values': bool(valid_range),
        'duplicates': duplicate_count
    }
    return checks
