import pandas as pd

def load_data(path='/content/sample_data/california_housing_train.csv'):
    """Loads data and performs basic validation."""
    df = pd.read_csv(path)
    return df

def validate_data(df):
    """Performs automated checks (Requirement a)."""
    # 1. Check for missing values
    missing_count = df.isnull().sum().sum()
    
    # 2. Check value range (e.g., median_house_value should be positive)
    valid_range = (df['median_house_value'] > 0).all()
    
    # 3. Check for duplicates
    duplicate_count = df.duplicated().sum()
    
    checks = {
        'missing_values': missing_count,
        'positive_prices': valid_range,
        'duplicates': duplicate_count
    }
    return checks
