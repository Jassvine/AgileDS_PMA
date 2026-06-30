import sys
import os
import pandas as pd

# Add src folder to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import load_data, validate_data

def test_data_quality():
    """Test suite for automated data validation."""
    # Create a dummy dataframe for testing if file doesn't exist in CI environment
    data = {'median_house_value': [100, 200, 300], 'other': [1, 2, 2]}
    df = pd.DataFrame(data)
    
    results = validate_data(df)
    
    # Assertions for CI/CD pipeline
    assert results['missing_values'] == 0, "Data contains missing values!"
    assert results['positive_prices'] == True, "Found negative house values!"
    assert results['duplicates'] == 0, "Found duplicate rows!"
