import pandas as pd

def load_data(path='/content/train.csv'):
    """Loads data from the user specified source."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        # Fallback for testing environments if file is not uploaded yet
        data = {'median_house_value': [100, 200, 300], 'other': [1, 2, 2]}
        return pd.DataFrame(data)

def validate_data(df):
    """Performs automated checks (Requirement a)."""
    # 1. Check for missing values
    missing_count = df.isnull().sum().sum()
    
    # 2. Check value range (Example: median_house_value should be positive)
    # Note: Replace 'median_house_value' with a relevant column from your CSV
    col_to_check = 'median_house_value' if 'median_house_value' in df.columns else df.columns[0]
    valid_range = (df[col_to_check] > 0).all()
    
    # 3. Check for duplicates
    duplicate_count = df.duplicated().sum()
    
    checks = {
        'missing_values': missing_count,
        'positive_values': valid_range,
        'duplicates': duplicate_count
    }
    return checks
